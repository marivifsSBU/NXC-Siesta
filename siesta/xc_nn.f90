! xc_nn.f90
!
!======================================================================
! SiestaXC neural-network XC (NNXC) engine
!
! This module implements the low-level NN forward pass for the GGA
! enhancement factors Fx and Fc, plus analytic derivatives with respect
! to the NN feature inputs (x0,x1,x2).
!
! Important implementation note (GELU):
!   The JAX reference implementation uses `jax.nn.gelu` with the default
!   `approximate=True` in JAX 0.7.0, i.e. the tanh-approx GELU.
!   To obtain bitwise-consistent results between Fortran and JAX, the
!   Fortran `gelu` and `dgelu` below use the tanh-approximation:
!     gelu(x) = 0.5*x*(1 + tanh( sqrt(2/pi)*(x + 0.044715*x^3) ))
!   and its analytic derivative.
!
! Debug support:
!   NNXC_DEBUG_RAW (below) can be toggled temporarily to print the raw
!   last-layer output `y` (pre-LOB) at a specific x2 value. It should
!   remain `.false.` for production builds.
!
!======================================================================
module xc_nn
  implicit none
  private
  public :: load_xc_nn_weights
  public :: load_xc_nn_weights_nofdf
  public :: fx_forward_and_grad3, fc_forward_and_grad3
  public :: model_fx_from_rho_grad, model_fc_from_rho_grad
  public :: gelu, dgelu

  integer, parameter :: dp = kind(1.0d0)
  integer, save :: debug_count_fc = 0
  integer, parameter :: NNXC_MODE_TL = 1, NNXC_MODE_UNCT = 2
  integer, save :: FX_MODE = NNXC_MODE_TL
  integer, save :: FC_MODE = NNXC_MODE_TL

  ! NOTE: keep .false. for production; set .true. only during debugging.
  ! Debug controls (set to .true. temporarily when diagnosing JAX/Fortran mismatches)
  logical, parameter :: NNXC_DEBUG_RAW = .false.
  real(dp), parameter :: NNXC_DEBUG_X2_TARGET = 1.336560450412419_dp
  real(dp), parameter :: NNXC_DEBUG_TOL = 1.0d-12
  logical, save :: nnxc_printed_fx_raw = .false.
  logical, save :: nnxc_printed_fc_raw = .false.

  ! Network architecture (can be overridden via the .fdf input).
  ! We compile with MAX sizes and use FX_IN/FX_H/FC_IN/FC_H at runtime.
  integer, parameter :: FX_IN_MAX = 3, FX_H_MAX = 256, FX_OUT = 1
  integer, parameter :: FC_IN_MAX = 3, FC_H_MAX = 256, FC_OUT = 1

  ! Defaults (match the common notebook/exporter setup):
  !   Fx: in=1 (uses x2 only), hidden=16
  !   Fc: in=3 (uses x0,x1,x2), hidden=16
  integer, save :: FX_IN = 1, FX_H = 16
  integer, save :: FC_IN = 3, FC_H = 16

  ! Weights and biases for Fx: shapes match numpy (out, in)
  real(dp), save :: W_fx1(FX_H_MAX, FX_IN_MAX), b_fx1(FX_H_MAX)
  real(dp), save :: W_fx2(FX_H_MAX, FX_H_MAX),  b_fx2(FX_H_MAX)
  real(dp), save :: W_fx3(FX_H_MAX, FX_H_MAX),  b_fx3(FX_H_MAX)
  real(dp), save :: W_fx4(FX_OUT,  FX_H_MAX),   b_fx4(FX_OUT)

  ! Weights and biases for Fc
  real(dp), save :: W_fc1(FC_H_MAX, FC_IN_MAX), b_fc1(FC_H_MAX)
  real(dp), save :: W_fc2(FC_H_MAX, FC_H_MAX),  b_fc2(FC_H_MAX)
  real(dp), save :: W_fc3(FC_H_MAX, FC_H_MAX),  b_fc3(FC_H_MAX)
  real(dp), save :: W_fc4(FC_OUT,  FC_H_MAX),   b_fc4(FC_OUT)

  ! Lieb–Oxford limits (must match your Python models)
  real(dp), parameter :: LOB_FX = 1.804_dp
  real(dp), parameter :: LOB_FC = 2.0_dp

contains

  !---------------- GELU and its derivative ----------------

  pure function gelu(x) result(y)
    real(dp), intent(in) :: x
    real(dp) :: y
    real(dp), parameter :: a = 0.79788456080286535588_dp  ! sqrt(2/pi)
    real(dp), parameter :: c = 0.044715_dp
    real(dp) :: u

    u = a * (x + c * x*x*x)
    y = 0.5_dp * x * (1.0_dp + tanh(u))
  end function gelu

  pure function dgelu(x) result(dy)
    real(dp), intent(in) :: x
    real(dp) :: dy
    real(dp), parameter :: a = 0.79788456080286535588_dp  ! sqrt(2/pi)
    real(dp), parameter :: c = 0.044715_dp
    real(dp) :: u, t, du_dx

    u = a * (x + c * x*x*x)
    t = tanh(u)
    du_dx = a * (1.0_dp + 3.0_dp*c*x*x)

    dy = 0.5_dp * (1.0_dp + t) + 0.5_dp * x * (1.0_dp - t*t) * du_dx
  end function dgelu

  !---------------- LOB and its derivative -----------------

  pure function lob(x, limit) result(y)
    real(dp), intent(in) :: x, limit
    real(dp) :: y
    real(dp) :: u, s
    u = x - log(limit - 1.0_dp)
    s = 1.0_dp / (1.0_dp + exp(-u))   ! sigmoid(u)
    y = limit * s - 1.0_dp
  end function lob

  pure function dlob(x, limit) result(dy)
    real(dp), intent(in) :: x, limit
    real(dp) :: dy
    real(dp) :: u, s
    u = x - log(limit - 1.0_dp)
    s = 1.0_dp / (1.0_dp + exp(-u))
    dy = limit * s * (1.0_dp - s)
  end function dlob

  !---------------- I/O helper -----------------

  subroutine read_matrix(filename, A)
    character(len=*), intent(in) :: filename
    real(dp), intent(out) :: A(:,:)
    integer :: iostat, i, j
    open(unit=77, file=filename, status='old', action='read', iostat=iostat)
    if (iostat /= 0) then
       write(*,*) "Error opening ", trim(filename), " iostat=", iostat
       stop
    end if
    ! Read matrix in row-major order to match numpy's default
    read(77, *, iostat=iostat) ((A(i,j), j=1,size(A,2)), i=1,size(A,1))
    if (iostat /= 0) then
       write(*,*) "Error reading ", trim(filename), " iostat=", iostat, &
                  " expected shape=(", size(A,1), ",", size(A,2), ")"
       stop
    end if
    close(77)
  end subroutine read_matrix

  subroutine read_vector(filename, v)
    character(len=*), intent(in) :: filename
    real(dp), intent(out) :: v(:)
    integer :: iostat
    open(unit=78, file=filename, status='old', action='read', iostat=iostat)
    if (iostat /= 0) then
       write(*,*) "Error opening ", trim(filename), " iostat=", iostat
       stop
    end if
    read(78, *, iostat=iostat) v
    if (iostat /= 0) then
       write(*,*) "Error reading ", trim(filename), " iostat=", iostat, &
                  " expected length=", size(v)
       stop
    end if
    close(78)
  end subroutine read_vector

  !---------------- Public loader -----------------

  subroutine load_xc_nn_weights(prefix)
    use fdf, only: fdf_string
    ! prefix is the same --out-prefix used in the exporter (e.g. 'xc_nn')
    character(len=*), intent(in) :: prefix
    character(len=256) :: fname
    character(len=256) :: sval
    integer :: ios
    integer :: fx_in_new, fx_h_new, fc_in_new, fc_h_new
    integer :: fx_mode_new, fc_mode_new

    ! Optional overrides from the .fdf file:
    !   NNXC.FX_IN, NNXC.FX_H, NNXC.FC_IN, NNXC.FC_H
    !   NNXC.H sets both hidden sizes (shorthand)
    fx_in_new = FX_IN
    fx_h_new  = FX_H
    fc_in_new = FC_IN
    fc_h_new  = FC_H
    fx_mode_new = FX_MODE
    fc_mode_new = FC_MODE

    sval = fdf_string('NNXC.H', '')
    if (len_trim(sval) > 0) then
       read(sval, *, iostat=ios) fx_h_new
       if (ios /= 0) then
          write(*,*) "Error parsing NNXC.H = ", trim(sval)
          stop
       end if
       fc_h_new = fx_h_new
    end if

    sval = fdf_string('NNXC.FX_IN', '')
    if (len_trim(sval) > 0) then
       read(sval, *, iostat=ios) fx_in_new
       if (ios /= 0) then
          write(*,*) "Error parsing NNXC.FX_IN = ", trim(sval)
          stop
       end if
    end if

    sval = fdf_string('NNXC.FX_H', '')
    if (len_trim(sval) > 0) then
       read(sval, *, iostat=ios) fx_h_new
       if (ios /= 0) then
          write(*,*) "Error parsing NNXC.FX_H = ", trim(sval)
          stop
       end if
    end if

    sval = fdf_string('NNXC.FC_IN', '')
    if (len_trim(sval) > 0) then
       read(sval, *, iostat=ios) fc_in_new
       if (ios /= 0) then
          write(*,*) "Error parsing NNXC.FC_IN = ", trim(sval)
          stop
       end if
    end if

    sval = fdf_string('NNXC.FC_H', '')
    if (len_trim(sval) > 0) then
       read(sval, *, iostat=ios) fc_h_new
       if (ios /= 0) then
          write(*,*) "Error parsing NNXC.FC_H = ", trim(sval)
          stop
       end if
    end if

    sval = fdf_string('NNXC.FX_MODE', '')
    if (len_trim(sval) > 0) then
       select case (trim(adjustl(sval)))
       case ('TL','tl','Tl','tL')
          fx_mode_new = NNXC_MODE_TL
       case ('UNCT','unct','Unct','uNCT','unCt','uncT')
          fx_mode_new = NNXC_MODE_UNCT
       case default
          write(*,*) "Error parsing NNXC.FX_MODE = ", trim(sval), " (supported: TL or UNCT)"
          stop
       end select
    end if

    sval = fdf_string('NNXC.FC_MODE', '')
    if (len_trim(sval) > 0) then
       select case (trim(adjustl(sval)))
       case ('TL','tl','Tl','tL')
          fc_mode_new = NNXC_MODE_TL
       case ('UNCT','unct','Unct','uNCT','unCt','uncT')
          fc_mode_new = NNXC_MODE_UNCT
       case default
          write(*,*) "Error parsing NNXC.FC_MODE = ", trim(sval), " (supported: TL or UNCT)"
          stop
       end select
    end if

    if (.not.(fx_in_new == 1 .or. fx_in_new == 3)) then
       write(*,*) "NNXC: unsupported FX_IN=", fx_in_new, " (supported: 1 or 3)"
       stop
    end if
    if (.not.(fc_in_new == 1 .or. fc_in_new == 3)) then
       write(*,*) "NNXC: unsupported FC_IN=", fc_in_new, " (supported: 1 or 3)"
       stop
    end if
    if (fx_h_new <= 0 .or. fc_h_new <= 0) then
       write(*,*) "NNXC: hidden sizes must be > 0. FX_H=", fx_h_new, " FC_H=", fc_h_new
       stop
    end if
    if (fx_in_new > FX_IN_MAX .or. fx_h_new > FX_H_MAX .or. fc_in_new > FC_IN_MAX .or. fc_h_new > FC_H_MAX) then
       write(*,*) "NNXC: requested dims exceed compiled MAX. Requested: ", &
                  "FX_IN=", fx_in_new, " FX_H=", fx_h_new, " FC_IN=", fc_in_new, " FC_H=", fc_h_new
       write(*,*) "NNXC: compiled MAX: FX_IN_MAX=", FX_IN_MAX, " FX_H_MAX=", FX_H_MAX, &
                  " FC_IN_MAX=", FC_IN_MAX, " FC_H_MAX=", FC_H_MAX
       stop
    end if

    FX_IN = fx_in_new
    FX_H  = fx_h_new
    FC_IN = fc_in_new
    FC_H  = fc_h_new
    FX_MODE = fx_mode_new
    FC_MODE = fc_mode_new

    write(*,*) "NNXC architecture: FX_IN=", FX_IN, " FX_H=", FX_H, " FC_IN=", FC_IN, " FC_H=", FC_H, &
               " FX_MODE=", FX_MODE, " FC_MODE=", FC_MODE

    ! Fx
    write(fname,'(a,"_fx_W1.dat")') trim(prefix)
    call read_matrix(fname, W_fx1(1:FX_H, 1:FX_IN))
    write(fname,'(a,"_fx_b1.dat")') trim(prefix)
    call read_vector(fname, b_fx1(1:FX_H))

    write(fname,'(a,"_fx_W2.dat")') trim(prefix)
    call read_matrix(fname, W_fx2(1:FX_H, 1:FX_H))
    write(fname,'(a,"_fx_b2.dat")') trim(prefix)
    call read_vector(fname, b_fx2(1:FX_H))

    write(fname,'(a,"_fx_W3.dat")') trim(prefix)
    call read_matrix(fname, W_fx3(1:FX_H, 1:FX_H))
    write(fname,'(a,"_fx_b3.dat")') trim(prefix)
    call read_vector(fname, b_fx3(1:FX_H))

    write(fname,'(a,"_fx_W4.dat")') trim(prefix)
    call read_matrix(fname, W_fx4(1:FX_OUT, 1:FX_H))
    write(fname,'(a,"_fx_b4.dat")') trim(prefix)
    call read_vector(fname, b_fx4)

    ! Fc
    write(fname,'(a,"_fc_W1.dat")') trim(prefix)
    call read_matrix(fname, W_fc1(1:FC_H, 1:FC_IN))
    write(fname,'(a,"_fc_b1.dat")') trim(prefix)
    call read_vector(fname, b_fc1(1:FC_H))

    write(fname,'(a,"_fc_W2.dat")') trim(prefix)
    call read_matrix(fname, W_fc2(1:FC_H, 1:FC_H))
    write(fname,'(a,"_fc_b2.dat")') trim(prefix)
    call read_vector(fname, b_fc2(1:FC_H))

    write(fname,'(a,"_fc_W3.dat")') trim(prefix)
    call read_matrix(fname, W_fc3(1:FC_H, 1:FC_H))
    write(fname,'(a,"_fc_b3.dat")') trim(prefix)
    call read_vector(fname, b_fc3(1:FC_H))

    write(fname,'(a,"_fc_W4.dat")') trim(prefix)
    call read_matrix(fname, W_fc4(1:FC_OUT, 1:FC_H))
    write(fname,'(a,"_fc_b4.dat")') trim(prefix)
    call read_vector(fname, b_fc4)

    write(*,*) "Loaded XC NN weights with prefix ", trim(prefix)
  end subroutine load_xc_nn_weights

  !---------------- Public loader (no FDF required) -----------------

  subroutine load_xc_nn_weights_nofdf(prefix)
    ! Standalone/test helper: load weights using current FX_IN/FX_H/FC_IN/FC_H
    ! without touching the FDF subsystem.
    character(len=*), intent(in) :: prefix
    character(len=256) :: fname

    if (.not.(FX_IN == 1 .or. FX_IN == 3)) then
       write(*,*) "NNXC: unsupported FX_IN=", FX_IN, " (supported: 1 or 3)"
       stop
    end if
    if (.not.(FC_IN == 1 .or. FC_IN == 3)) then
       write(*,*) "NNXC: unsupported FC_IN=", FC_IN, " (supported: 1 or 3)"
       stop
    end if
    if (FX_H <= 0 .or. FC_H <= 0) then
       write(*,*) "NNXC: hidden sizes must be > 0. FX_H=", FX_H, " FC_H=", FC_H
       stop
    end if
    if (FX_IN > FX_IN_MAX .or. FX_H > FX_H_MAX .or. FC_IN > FC_IN_MAX .or. FC_H > FC_H_MAX) then
       write(*,*) "NNXC: requested dims exceed compiled MAX. Requested: ", &
                  "FX_IN=", FX_IN, " FX_H=", FX_H, " FC_IN=", FC_IN, " FC_H=", FC_H
       write(*,*) "NNXC: compiled MAX: FX_IN_MAX=", FX_IN_MAX, " FX_H_MAX=", FX_H_MAX, &
                  " FC_IN_MAX=", FC_IN_MAX, " FC_H_MAX=", FC_H_MAX
       stop
    end if

    write(*,*) "NNXC architecture (noFDF): FX_IN=", FX_IN, " FX_H=", FX_H, " FC_IN=", FC_IN, " FC_H=", FC_H

    ! Fx
    write(fname,'(a,"_fx_W1.dat")') trim(prefix)
    call read_matrix(fname, W_fx1(1:FX_H, 1:FX_IN))
    write(fname,'(a,"_fx_b1.dat")') trim(prefix)
    call read_vector(fname, b_fx1(1:FX_H))

    write(fname,'(a,"_fx_W2.dat")') trim(prefix)
    call read_matrix(fname, W_fx2(1:FX_H, 1:FX_H))
    write(fname,'(a,"_fx_b2.dat")') trim(prefix)
    call read_vector(fname, b_fx2(1:FX_H))

    write(fname,'(a,"_fx_W3.dat")') trim(prefix)
    call read_matrix(fname, W_fx3(1:FX_H, 1:FX_H))
    write(fname,'(a,"_fx_b3.dat")') trim(prefix)
    call read_vector(fname, b_fx3(1:FX_H))

    write(fname,'(a,"_fx_W4.dat")') trim(prefix)
    call read_matrix(fname, W_fx4(1:FX_OUT, 1:FX_H))
    write(fname,'(a,"_fx_b4.dat")') trim(prefix)
    call read_vector(fname, b_fx4)

    ! Fc
    write(fname,'(a,"_fc_W1.dat")') trim(prefix)
    call read_matrix(fname, W_fc1(1:FC_H, 1:FC_IN))
    write(fname,'(a,"_fc_b1.dat")') trim(prefix)
    call read_vector(fname, b_fc1(1:FC_H))

    write(fname,'(a,"_fc_W2.dat")') trim(prefix)
    call read_matrix(fname, W_fc2(1:FC_H, 1:FC_H))
    write(fname,'(a,"_fc_b2.dat")') trim(prefix)
    call read_vector(fname, b_fc2(1:FC_H))

    write(fname,'(a,"_fc_W3.dat")') trim(prefix)
    call read_matrix(fname, W_fc3(1:FC_H, 1:FC_H))
    write(fname,'(a,"_fc_b3.dat")') trim(prefix)
    call read_vector(fname, b_fc3(1:FC_H))

    write(fname,'(a,"_fc_W4.dat")') trim(prefix)
    call read_matrix(fname, W_fc4(1:FC_OUT, 1:FC_H))
    write(fname,'(a,"_fc_b4.dat")') trim(prefix)
    call read_vector(fname, b_fc4)

    write(*,*) "Loaded XC NN weights with prefix (noFDF) ", trim(prefix)
  end subroutine load_xc_nn_weights_nofdf

  !---------------- Fx MLP + LOB: forward + d/dx0,d/dx1,d/dx2 -----------------

  subroutine fx_forward_and_grad3(x0, x1, x2, Fx, dFx_dx0, dFx_dx1, dFx_dx2)
    ! Inputs:  x0, x1, x2
    ! Output:  TL   mode: Fx = 1 + LOB(x2 * MLP_x(...))
    !          UNCT mode: Fx = MLP_x(...)
    !          dFx_dx0, dFx_dx1, dFx_dx2
    real(dp), intent(in)  :: x0, x1, x2
    real(dp), intent(out) :: Fx, dFx_dx0, dFx_dx1, dFx_dx2

    real(dp) :: a0(FX_IN_MAX)
    real(dp) :: z1(FX_H_MAX), a1(FX_H_MAX)
    real(dp) :: z2(FX_H_MAX), a2(FX_H_MAX)
    real(dp) :: z3(FX_H_MAX), a3(FX_H_MAX)
    real(dp) :: z4(FX_OUT)
    real(dp) :: y
    real(dp) :: delta3(FX_H_MAX), delta2(FX_H_MAX), delta1(FX_H_MAX)
    real(dp) :: grad_in(FX_IN_MAX)
    real(dp) :: zlob, dlobdz
    integer :: i, j

    ! Forward: map (x0,x1,x2) -> NN inputs
    a0 = 0.0_dp
    if (FX_IN == 1) then
       a0(1) = x2
    else if (FX_IN == 3) then
       a0(1) = x0
       a0(2) = x1
       a0(3) = x2
    else
       write(*,*) "NNXC: unsupported FX_IN=", FX_IN, " (supported: 1 or 3)"
       stop
    end if

    do i = 1, FX_H
       z1(i) = b_fx1(i)
       do j = 1, FX_IN
          z1(i) = z1(i) + W_fx1(i,j) * a0(j)
       end do
       a1(i) = gelu(z1(i))
    end do

    do i = 1, FX_H
       z2(i) = b_fx2(i)
       do j = 1, FX_H
          z2(i) = z2(i) + W_fx2(i,j) * a1(j)
       end do
       a2(i) = gelu(z2(i))
    end do

    do i = 1, FX_H
       z3(i) = b_fx3(i)
       do j = 1, FX_H
          z3(i) = z3(i) + W_fx3(i,j) * a2(j)
       end do
       a3(i) = gelu(z3(i))
    end do

    do i = 1, FX_OUT
       z4(i) = b_fx4(i)
       do j = 1, FX_H
          z4(i) = z4(i) + W_fx4(i,j) * a3(j)
       end do
    end do
    y = z4(1)

    if (NNXC_DEBUG_RAW) then
       if ((.not. nnxc_printed_fx_raw) .and. (abs(x2 - NNXC_DEBUG_X2_TARGET) < NNXC_DEBUG_TOL)) then
          write(*,'("FX raw y=",1PE24.16,"  x0=",1PE24.16," x1=",1PE24.16," x2=",1PE24.16)') y, x0, x1, x2
          nnxc_printed_fx_raw = .true.
       end if
    end if

    select case (FX_MODE)
    case (NNXC_MODE_TL)
       ! Apply LOB: Fx = 1 + lob(x2 * y, LOB_FX)
       zlob = x2 * y
       Fx   = 1.0_dp + lob(zlob, LOB_FX)
    case (NNXC_MODE_UNCT)
       Fx = y
    case default
       write(*,*) "NNXC: unsupported FX_MODE=", FX_MODE
       stop
    end select

    ! Backprop to get dy/dx0,dy/dx1,dy/dx2
    do i = 1, FX_H
       delta3(i) = 0.0_dp
       do j = 1, FX_OUT
          delta3(i) = delta3(i) + W_fx4(j,i) * 1.0_dp
       end do
       delta3(i) = delta3(i) * dgelu(z3(i))
    end do

    do i = 1, FX_H
       delta2(i) = 0.0_dp
       do j = 1, FX_H
          delta2(i) = delta2(i) + W_fx3(j,i) * delta3(j)
       end do
       delta2(i) = delta2(i) * dgelu(z2(i))
    end do

    do i = 1, FX_H
       delta1(i) = 0.0_dp
       do j = 1, FX_H
          delta1(i) = delta1(i) + W_fx2(j,i) * delta2(j)
       end do
       delta1(i) = delta1(i) * dgelu(z1(i))
    end do

    grad_in = 0.0_dp
    do j = 1, FX_IN
       do i = 1, FX_H
          grad_in(j) = grad_in(j) + W_fx1(i,j) * delta1(i)
       end do
    end do

    select case (FX_MODE)
    case (NNXC_MODE_TL)
       dlobdz  = dlob(zlob, LOB_FX)
       ! Fx = 1 + LOB(x2 * y)
       if (FX_IN == 1) then
          dFx_dx0 = 0.0_dp
          dFx_dx1 = 0.0_dp
          dFx_dx2 = dlobdz * ( y  + x2 * grad_in(1) )
       else if (FX_IN == 3) then
          dFx_dx0 = dlobdz * ( x2 * grad_in(1) )
          dFx_dx1 = dlobdz * ( x2 * grad_in(2) )
          dFx_dx2 = dlobdz * ( y  + x2 * grad_in(3) )
       else
          write(*,*) "NNXC: unsupported FX_IN=", FX_IN, " (supported: 1 or 3)"
          stop
       end if
    case (NNXC_MODE_UNCT)
       if (FX_IN == 1) then
          dFx_dx0 = 0.0_dp
          dFx_dx1 = 0.0_dp
          dFx_dx2 = grad_in(1)
       else if (FX_IN == 3) then
          dFx_dx0 = grad_in(1)
          dFx_dx1 = grad_in(2)
          dFx_dx2 = grad_in(3)
       else
          write(*,*) "NNXC: unsupported FX_IN=", FX_IN, " (supported: 1 or 3)"
          stop
       end if
    case default
       write(*,*) "NNXC: unsupported FX_MODE=", FX_MODE
       stop
    end select

  end subroutine fx_forward_and_grad3

  !---------------- Fc MLP + LOB: forward + d/dx0,d/dx1,d/dx2 -----------------

  subroutine fc_forward_and_grad3(x0, x1, x2, Fc, dFc_dx0, dFc_dx1, dFc_dx2)
    ! Inputs:  x0, x1, x2
    ! Output:  TL   mode: Fc = 1 + LOB(x2 * MLP_c(...))
    !          UNCT mode: Fc = MLP_c(...)
    !          dFc_dx0, dFc_dx1, dFc_dx2
    real(dp), intent(in)  :: x0, x1, x2
    real(dp), intent(out) :: Fc, dFc_dx0, dFc_dx1, dFc_dx2

    real(dp) :: a0(FC_IN_MAX)
    real(dp) :: z1(FC_H_MAX), a1(FC_H_MAX)
    real(dp) :: z2(FC_H_MAX), a2(FC_H_MAX)
    real(dp) :: z3(FC_H_MAX), a3(FC_H_MAX)
    real(dp) :: z4(FC_OUT)
    real(dp) :: y
    real(dp) :: delta3(FC_H_MAX), delta2(FC_H_MAX), delta1(FC_H_MAX)
    real(dp) :: grad_in(FC_IN_MAX)
    real(dp) :: zlob, dlobdz
    integer :: i, j

    ! Forward: map (x0,x1,x2) -> NN inputs
    a0 = 0.0_dp
    if (FC_IN == 1) then
       a0(1) = x2
    else if (FC_IN == 3) then
       a0(1) = x0
       a0(2) = x1
       a0(3) = x2
    else
       write(*,*) "NNXC: unsupported FC_IN=", FC_IN, " (supported: 1 or 3)"
       stop
    end if

    do i = 1, FC_H
       z1(i) = b_fc1(i)
       do j = 1, FC_IN
          z1(i) = z1(i) + W_fc1(i,j) * a0(j)
       end do
       a1(i) = gelu(z1(i))
    end do

    do i = 1, FC_H
       z2(i) = b_fc2(i)
       do j = 1, FC_H
          z2(i) = z2(i) + W_fc2(i,j) * a1(j)
       end do
       a2(i) = gelu(z2(i))
    end do

    do i = 1, FC_H
       z3(i) = b_fc3(i)
       do j = 1, FC_H
          z3(i) = z3(i) + W_fc3(i,j) * a2(j)
       end do
       a3(i) = gelu(z3(i))
    end do

    do i = 1, FC_OUT
       z4(i) = b_fc4(i)
       do j = 1, FC_H
          z4(i) = z4(i) + W_fc4(i,j) * a3(j)
       end do
    end do
    y = z4(1)

    if (NNXC_DEBUG_RAW) then
       if ((.not. nnxc_printed_fc_raw) .and. (abs(x2 - NNXC_DEBUG_X2_TARGET) < NNXC_DEBUG_TOL)) then
          write(*,'("FC raw y=",1PE24.16,"  x0=",1PE24.16," x1=",1PE24.16," x2=",1PE24.16)') y, x0, x1, x2
          nnxc_printed_fc_raw = .true.
       end if
    end if

    select case (FC_MODE)
    case (NNXC_MODE_TL)
       zlob = x2 * y
       Fc   = 1.0_dp + lob(zlob, LOB_FC)
    case (NNXC_MODE_UNCT)
       Fc = y
    case default
       write(*,*) "NNXC: unsupported FC_MODE=", FC_MODE
       stop
    end select

    ! Backprop to get dy/dx0, dy/dx1, dy/dx2
    do i = 1, FC_H
       delta3(i) = 0.0_dp
       do j = 1, FC_OUT
          delta3(i) = delta3(i) + W_fc4(j,i) * 1.0_dp
       end do
       delta3(i) = delta3(i) * dgelu(z3(i))
    end do

    do i = 1, FC_H
       delta2(i) = 0.0_dp
       do j = 1, FC_H
          delta2(i) = delta2(i) + W_fc3(j,i) * delta3(j)
       end do
       delta2(i) = delta2(i) * dgelu(z2(i))
    end do

    do i = 1, FC_H
       delta1(i) = 0.0_dp
       do j = 1, FC_H
          delta1(i) = delta1(i) + W_fc2(j,i) * delta2(j)
       end do
       delta1(i) = delta1(i) * dgelu(z1(i))
    end do

    grad_in = 0.0_dp
    do j = 1, FC_IN
       do i = 1, FC_H
          grad_in(j) = grad_in(j) + W_fc1(i,j) * delta1(i)
       end do
    end do

    select case (FC_MODE)
    case (NNXC_MODE_TL)
       dlobdz   = dlob(zlob, LOB_FC)
       ! Fc = 1 + LOB(x2 * y)
       if (FC_IN == 1) then
          dFc_dx0 = 0.0_dp
          dFc_dx1 = 0.0_dp
          dFc_dx2 = dlobdz * ( y  + x2 * grad_in(1) )
       else if (FC_IN == 3) then
          dFc_dx0 = dlobdz * ( x2 * grad_in(1) )
          dFc_dx1 = dlobdz * ( x2 * grad_in(2) )
          dFc_dx2 = dlobdz * ( y  + x2 * grad_in(3) )
       else
          write(*,*) "NNXC: unsupported FC_IN=", FC_IN, " (supported: 1 or 3)"
          stop
       end if
    case (NNXC_MODE_UNCT)
       if (FC_IN == 1) then
          dFc_dx0 = 0.0_dp
          dFc_dx1 = 0.0_dp
          dFc_dx2 = grad_in(1)
       else if (FC_IN == 3) then
          dFc_dx0 = grad_in(1)
          dFc_dx1 = grad_in(2)
          dFc_dx2 = grad_in(3)
       else
          write(*,*) "NNXC: unsupported FC_IN=", FC_IN, " (supported: 1 or 3)"
          stop
       end if
    case default
       write(*,*) "NNXC: unsupported FC_MODE=", FC_MODE
       stop
    end select

    ! Debug print disabled
    debug_count_fc = debug_count_fc + 1

  end subroutine fc_forward_and_grad3


  !---------------- Top-level model wrappers: (rho,grad_rho) -> Fx,Fc -----------------
  !
  ! Legacy wrappers (rho,|grad rho|) -> Fx/Fc
  ! --------------------------------------
  ! These wrappers implement an older feature convention (log10-based)
  ! and are retained for backwards compatibility and ad hoc testing.
  !
  ! The production SIESTA path uses `xc_nn_siesta.f90` (NNGGAXC), which
  ! constructs features with natural logs and the spin-scaling / zeta'
  ! definitions validated against the JAX reference.
  !
  ! If you are debugging JAX↔Fortran consistency, prefer the NNGGAXC path
  ! (or the standalone debug testers), not these wrappers.
  !

  subroutine model_fx_from_rho_grad(rho, grad_rho, Fx)
    !! Fortran version of GGA_Fx_G_transf_lin.__call__(inputs=[rho, grad_rho])
    implicit none
    real(dp), intent(in)  :: rho, grad_rho
    real(dp), intent(out) :: Fx
    real(dp) :: kF, s, s2, x0, x1, x2
    real(dp) :: dFx_dx0, dFx_dx1, dFx_dx2
    real(dp), parameter :: pi = acos(-1.0_dp)

    if (rho <= 0.0_dp) then
       Fx = 1.0_dp
       return
    end if

    ! k_F = (3 * pi^2 * rho)^(1/3)
    kF = (3.0_dp * pi*pi * rho)**(1.0_dp/3.0_dp)
    ! s = |grad rho| / (2 k_F rho)
    s  = grad_rho / (2.0_dp * kF * rho)
    s2 = s*s

    ! x2 = log10(1 + s) * (1 - exp(-s^2))  (legacy)
    x2 = log10(1.0_dp + s) * (1.0_dp - exp(-s2))

    ! x0 = log10(rho^(1/3) + eps)  (legacy)
    x0 = log10(rho**(1.0_dp/3.0_dp) + 1.0d-5)

    ! Unpolarized spin feature: zeta' = 1 => x1 = log10(1 + eps)  (legacy)
    x1 = log10(1.0_dp + 1.0d-5)

    call fx_forward_and_grad3(x0, x1, x2, Fx, dFx_dx0, dFx_dx1, dFx_dx2)
  end subroutine model_fx_from_rho_grad


  subroutine model_fc_from_rho_grad(rho, grad_rho, Fc)
    !! Fortran version of GGA_Fc_G_transf_lin.__call__(inputs=[rho, grad_rho])
    implicit none
    real(dp), intent(in)  :: rho, grad_rho
    real(dp), intent(out) :: Fc
    real(dp) :: kF, s, s2, x0, x1, x2
    real(dp) :: dFc_dx0, dFc_dx1, dFc_dx2
    real(dp), parameter :: eps = 1.0d-5
    real(dp), parameter :: pi  = acos(-1.0_dp)

    if (rho <= 0.0_dp) then
       Fc = 1.0_dp
       return
    end if

    ! k_F = (3 * pi^2 * rho)^(1/3)
    kF = (3.0_dp * pi*pi * rho)**(1.0_dp/3.0_dp)
    ! s = |grad rho| / (2 k_F rho)
    s  = grad_rho / (2.0_dp * kF * rho)
    s2 = s*s

    ! x2 = log10(1 + s) * (1 - exp(-s^2))  (legacy)
    x2 = log10(1.0_dp + s) * (1.0_dp - exp(-s2))

    ! x0 = log10(rho^(1/3) + eps)  (legacy)
    x0 = log10(rho**(1.0_dp/3.0_dp) + eps)

    ! Unpolarized spin feature: zeta' = 1 => x1 = log10(1 + eps)  (legacy)
    x1 = log10(1.0_dp + eps)

    call fc_forward_and_grad3(x0, x1, x2, Fc, dFc_dx0, dFc_dx1, dFc_dx2)
  end subroutine model_fc_from_rho_grad

end module xc_nn