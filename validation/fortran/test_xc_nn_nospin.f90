!======================================================================
! test_xc_nn.f90
!----------------------------------------------------------------------
! Standalone Fortran test program for the NN-GGA XC functional.
!
! PURPOSE
!   This code validates the Fortran implementation of the neural-network
!   GGA exchange--correlation functional used in SIESTA. It:
!
!     1) Loads NN parameters exported from JAX/Equinox models
!        (ASCII files xc_nn_*.dat).
!     2) Reads a real-space grid file with columns:
!          rho, grad_rho_x, grad_rho_y, grad_rho_z, weight
!     3) Evaluates the NN-enhanced LDA exchange and correlation energies
!        and their analytic derivatives at each grid point.
!     4) Integrates the XC energy:
!          E_xc = sum_i w_i * rho_i * eps_xc(rho_i, |grad rho_i|)
!
!   The program is intended as a developer / regression test to ensure
!   that:
!     - exported NN parameters are read correctly,
!     - the Fortran forward pass matches the Python/JAX implementation,
!     - analytic derivatives are consistent and finite.
!
! INPUT FILES
!   - NN parameter files in the current directory:
!       xc_nn_fx_W*.dat, xc_nn_fx_b*.dat
!       xc_nn_fc_W*.dat, xc_nn_fc_b*.dat
!
!   - A grid file (default: nn_rho_grad_w.dat) containing:
!       rho, gx, gy, gz, w
!     one grid point per line, in atomic units.
!
! OUTPUT
!   - Prints sample per-point values (first 10 points):
!       rho, |grad rho|, eps_xc, d( rho*eps_xc )/d rho, d( rho*eps_xc )/d|grad rho|
!   - Prints the integrated XC energy.
!
! NOTES
!   - This is NOT production SIESTA code.
!   - LDA reference derivatives here are computed numerically and are
!     used only for testing. In SIESTA, analytic EXCHNG and PW92C
!     routines are used instead.
!
! COMPILATION
!   From a directory containing xc_nn.f90:
!
!     gfortran xc_nn.f90 test_xc_nn.f90 -o test_xc_nn
!
!   On macOS, you may need to ensure the correct SDK is used, e.g.:
!
!     export SDKROOT=$(xcrun --sdk macosx --show-sdk-path)
!
! RUN
!     ./test_xc_nn
!
!======================================================================
program test_xc_nn
  use xc_nn           ! from xc_nn.f90
  implicit none
  integer, parameter :: dp = kind(1.0d0)

  character(len=*), parameter :: prefix = "xc_nn"
  character(len=*), parameter :: grid_file = "nn_rho_grad_w.dat"

  real(dp) :: rho, gx, gy, gz, w
  real(dp) :: grad_rho
  real(dp) :: ex, ec, exc, exc_int
  real(dp) :: dedrho, dedg
  integer :: ios, linecount
  character(len=256) :: line

  call load_xc_nn_weights(prefix)

  exc_int = 0.0_dp
  linecount = 0

  open(unit=10, file=grid_file, status='old', action='read', iostat=ios)
  if (ios /= 0) then
     write(*,*) "Error opening grid file ", trim(grid_file), " iostat=", ios
     stop
  end if

  do
     ! Read a whole line as text, skip headers or comments starting with '#'
     read(10, '(A)', iostat=ios) line
     if (ios /= 0) exit
     if (len_trim(line) == 0) cycle
     if (line(1:1) == '#') cycle

     ! Parse numeric data from the line
     read(line, *, iostat=ios) rho, gx, gy, gz, w
     if (ios /= 0) cycle

     linecount = linecount + 1

     if (rho <= 1.0d-14) cycle

     grad_rho = sqrt(gx*gx + gy*gy + gz*gz)

     ! Compute NN XC energies and derivatives
     call nn_xc_from_rho_grad(rho, gx, gy, gz, ex, ec, dedrho, dedg)
     exc = ex + ec

     if (linecount <= 10) then
        write(*,'("i=",I4," rho=",1PE12.5," grad=",1PE12.5, &
                  " exc=",1PE12.5," dedrho=",1PE12.5," dedg=",1PE12.5)') &
             linecount, rho, grad_rho, exc, dedrho, dedg
     end if

     ! Accumulate E_xc = ∑ w * ρ * exc
     exc_int = exc_int + w * rho * exc
  end do
  close(10)

  write(*,'("Read ",I10," grid points")') linecount
  write(*,'("Fortran NN-XC integrated E_xc = ", F20.12)') exc_int

contains

  subroutine nn_xc_from_rho_grad(rho_in, gx_in, gy_in, gz_in, ex, ec, dedrho, dedg)
    use xc_nn, only: fx_forward_and_grad, fc_forward_and_grad
    implicit none
    real(dp), intent(in)  :: rho_in, gx_in, gy_in, gz_in
    real(dp), intent(out) :: ex, ec, dedrho, dedg
    real(dp) :: rho, gx, gy, gz, g, tiny, pi, b_kF, kF, s, s2
    real(dp) :: epsx_LDA, epsc_LDA
    real(dp) :: Fx, dFx_dx1
    real(dp) :: Fc, dFc_dx0, dFc_dx1
    real(dp) :: f0, x0, dx0_drho
    real(dp) :: A, B, dA_ds, dB_ds, dx1_ds, x1
    real(dp) :: ds_drho, ds_dg, dx1_drho, dx1_dg
    real(dp) :: deLDA_drho_x, deLDA_drho_c
    real(dp) :: dr, rhop, rhom, fplus, fminus
    real(dp) :: dedrho_x, dedrho_c, dedg_x, dedg_c

    rho = rho_in
    gx  = gx_in
    gy  = gy_in
    gz  = gz_in

    tiny = 1.0d-14
    pi   = acos(-1.0_dp)
    b_kF = (3.0_dp*pi*pi)**(1.0_dp/3.0_dp)

    if (rho <= tiny) then
       ex     = 0.0_dp
       ec     = 0.0_dp
       dedrho = 0.0_dp
       dedg   = 0.0_dp
       return
    end if

    g = sqrt(gx*gx + gy*gy + gz*gz)

    ! LDA references
    epsx_LDA = eUEG_LDA_x(rho)
    epsc_LDA = eUEG_LDA_c_pw92_unpol(rho)

    ! Numerical derivative of (rho * epsx_LDA) wrt rho
    dr = 1.0d-5*rho
    if (dr < 1.0d-8) dr = 1.0d-8
    rhop  = rho + dr
    rhom  = max(rho - dr, tiny)
    fplus  = rhop * eUEG_LDA_x(rhop)
    fminus = rhom * eUEG_LDA_x(rhom)
    deLDA_drho_x = (fplus - fminus)/(rhop - rhom)

    ! Numerical derivative of (rho * epsc_LDA) wrt rho
    fplus  = rhop * eUEG_LDA_c_pw92_unpol(rhop)
    fminus = rhom * eUEG_LDA_c_pw92_unpol(rhom)
    deLDA_drho_c = (fplus - fminus)/(rhop - rhom)

    ! Fx,Fc features: kF, s, x0, x1 and their derivatives
    kF = b_kF * rho**(1.0_dp/3.0_dp)
    if (kF*rho > tiny) then
       s = g / (2.0_dp * kF * rho)
    else
       s = 0.0_dp
    end if
    s2 = s*s

    f0 = rho**(1.0_dp/3.0_dp) + 1.0d-5
    x0 = log10(f0)
    dx0_drho = (1.0_dp/log(10.0_dp)) * (1.0_dp/f0) * (1.0_dp/3.0_dp)*rho**(-2.0_dp/3.0_dp)

    A     = log10(1.0_dp + s)
    B     = 1.0_dp - exp(-s2)
    dA_ds = 1.0_dp / ((1.0_dp+s)*log(10.0_dp))
    dB_ds = 2.0_dp * s * exp(-s2)
    dx1_ds = dA_ds*B + A*dB_ds
    x1     = A*B

    if (rho > tiny) then
       ds_drho = -(4.0_dp/3.0_dp) * s / rho
    else
       ds_drho = 0.0_dp
    end if
    if (rho > tiny) then
       ds_dg = 1.0_dp / (2.0_dp * b_kF * rho**(4.0_dp/3.0_dp))
    else
       ds_dg = 0.0_dp
    end if

    dx1_drho = dx1_ds * ds_drho
    dx1_dg   = dx1_ds * ds_dg
    if (g < tiny) dx1_dg = 0.0_dp

    ! NN MLPs and their derivatives
    call fx_forward_and_grad( x1, Fx, dFx_dx1 )
    call fc_forward_and_grad( x0, x1, Fc, dFc_dx0, dFc_dx1 )

    ! Energies per electron
    ex = Fx * epsx_LDA
    ec = Fc * epsc_LDA

    ! Exchange contribution
    dedrho_x = Fx * deLDA_drho_x + rho*epsx_LDA * dFx_dx1 * dx1_drho
    dedg_x   = rho*epsx_LDA * dFx_dx1 * dx1_dg

    ! Correlation contribution
    dedrho_c = Fc * deLDA_drho_c + rho*epsc_LDA * &
         ( dFc_dx0*dx0_drho + dFc_dx1*dx1_drho )
    dedg_c   = rho*epsc_LDA * dFc_dx1 * dx1_dg

    dedrho = dedrho_x + dedrho_c
    dedg   = dedg_x   + dedg_c

  end subroutine nn_xc_from_rho_grad

  pure function eUEG_LDA_x(rho) result(epsx)
    real(dp), intent(in) :: rho
    real(dp) :: epsx
    real(dp) :: c
    if (rho <= 0.0_dp) then
       epsx = 0.0_dp
    else
       c = -0.75_dp * (3.0_dp/acos(-1.0_dp))**(1.0_dp/3.0_dp)
       epsx = c * rho**(1.0_dp/3.0_dp)
    end if
  end function eUEG_LDA_x

  ! Unpolarized PW92 correlation (same parameters as your Python eUEG_LDA_c)
  pure function eUEG_LDA_c_pw92_unpol(rho) result(epsc)
    real(dp), intent(in) :: rho
    real(dp) :: epsc
    real(dp) :: A(3), ALPHA1(3), BETA1(3), BETA2(3), BETA3(3), BETA4(3)
    real(dp) :: rs, B, C, G0, G1, G2
    integer :: k

    if (rho <= 0.0_dp) then
       epsc = 0.0_dp
       return
    end if

    ! From your Python code
    A      = (/ 0.031090690869654895_dp, 0.015545_dp, 0.016887_dp /)
    ALPHA1 = (/ 0.21370_dp, 0.20548_dp, 0.11125_dp /)
    BETA1  = (/ 7.5957_dp, 14.1189_dp, 10.357_dp /)
    BETA2  = (/ 3.5876_dp, 6.1977_dp, 3.6231_dp /)
    BETA3  = (/ 1.6382_dp, 3.3662_dp, 0.88026_dp /)
    BETA4  = (/ 0.49294_dp, 0.62517_dp, 0.49671_dp /)

    rs = (3.0_dp / (4.0_dp * acos(-1.0_dp) * rho))**(1.0_dp/3.0_dp)

    ! Only k=1 is used for unpolarized (zeta=0) in your Python version
    k = 1
    B = BETA1(k)*sqrt(rs) + BETA2(k)*rs + BETA3(k)*rs**1.5_dp + BETA4(k)*rs**2
    C = 1.0_dp + 1.0_dp / (2.0_dp*A(k)*B)
    epsc = -2.0_dp * A(k) * (1.0_dp + ALPHA1(k)*rs) * log(C)
  end function eUEG_LDA_c_pw92_unpol

end program test_xc_nn