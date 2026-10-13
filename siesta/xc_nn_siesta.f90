!======================================================================
! xc_nn_siesta.f90
!----------------------------------------------------------------------
! SIESTA-facing interface for the NN-based GGA XC functional (NNXC).
!
! This module implements:
!   - NNGGAXC: SIESTA callback returning (EX, EC) and first derivatives
!              of the *energy density* w.r.t. densities and density
!              gradients, in the format expected by SIESTA.
!   - NNXC_INIT: standalone/test initializer to load NN weights without
!                requiring the FDF subsystem.
!
! Conceptual note:
!   SIESTA computes the final XC potential on the real-space grid using
!   numerical divergence operators (Balbás & Soler, PRB 64, 165110).
!   Therefore, this routine returns only first derivatives of the energy
!   density with respect to its local inputs (rho, zeta, gradients).
!
! Numerical safety:
!   Reduced gradients involve divisions by rho and kF(rho). To avoid
!   NaNs at (or extremely near) vacuum points, we use a tiny additive
!   denominator in the reduced-gradient definitions.
!
!======================================================================
module xc_nn_siesta
  use precision, only: dp
  use m_ldaxc, only: exchng, pw92c
  use xc_nn
  use fdf, only: fdf_string
  implicit none
  private
  public :: NNGGAXC, NNXC_INIT

  logical, save :: nn_weights_loaded = .false.
  character(len=256), save :: nn_prefix = ''

contains

  !--------------------------------------------------------------------
  ! NNXC_INIT(prefix)
  !   Standalone/test helper: loads NN weights without requiring FDF.
  !   For production SIESTA runs, NNGGAXC will load weights from FDF.
  !--------------------------------------------------------------------
  subroutine NNXC_INIT(prefix)
    ! Standalone/test initializer: load NN weights without requiring FDF.
    implicit none
    character(len=*), intent(in) :: prefix

    nn_prefix = trim(prefix)
    call load_xc_nn_weights_nofdf(trim(nn_prefix))
    nn_weights_loaded = .true.
  end subroutine NNXC_INIT

  subroutine NNGGAXC( IREL, nSpin, Dens, GDens, &
                      EX, EC, dEXdD, dECdD, dEXdGD, dECdGD )
    implicit none
    integer, intent(in) :: IREL, nSpin
    real(dp), intent(in)  :: Dens(nSpin), GDens(3,nSpin)
    real(dp), intent(out) :: EX, EC
    real(dp), intent(out) :: dEXdD(nSpin), dECdD(nSpin)
    real(dp), intent(out) :: dEXdGD(3,nSpin), dECdGD(3,nSpin)

    integer :: is, k
    real(dp) :: rho, ru, rd, Gtot(3), g, tiny
    real(dp), parameter :: tiny_denom = 1.0d-30

    ! LDA references (exchange via EXCHNG, correlation via PW92C)
    real(dp) :: epsx_LDA, epsc_LDA
    real(dp) :: ex_den, ec_den
    real(dp) :: densx(1), exu_up, exu_dn, vxu_up(1), vxu_dn(1)
    real(dp) :: Dlda(2), vcunif(2), ecunif

    ! NN outputs and partial derivatives wrt NN inputs (x0,x1,x2)
    real(dp) :: Fx_u, dFx_dx0_u, dFx_dx1_u, dFx_dx2_u
    real(dp) :: Fx_d, dFx_dx0_d, dFx_dx1_d, dFx_dx2_d
    real(dp) :: Fc, dFc_dx0, dFc_dx1, dFc_dx2

    ! Feature definitions
    real(dp) :: pi, b_kF, kF
    real(dp) :: x0_u, x1_u, x2_u
    real(dp) :: x0_d, x1_d, x2_d
    real(dp) :: x0_c, x1_c, x2_c
    real(dp) :: dx0_drho_u, dx0_drho_d, dx0_drho_c
    real(dp) :: dx1_dru, dx1_drd
    real(dp) :: dx2_ds_u, dx2_ds_d, dx2_ds_c
    real(dp) :: dx2_drho_c, dx2_dg_c
    real(dp) :: ds_drho_u, ds_drho_d, ds_dg_u, ds_dg_d
    real(dp) :: ds_drho_c, ds_dg_c

    ! Additional local declarations
    real(dp) :: s
    real(dp) :: dx2_drho_u, dx2_drho_d
    real(dp) :: dFx_drho_u, dFx_drho_d
    real(dp) :: dFx_dg_u, dFx_dg_d

    ! Derivatives of energy density (per volume) wrt rho and |grad rho|
    real(dp) :: dedrho_x(2), dedrho_c(2)
    real(dp) :: dedg_x_u, dedg_x_d, dedg_c

    ! Gradients per spin
    real(dp) :: GradU(3), GradD(3)
    real(dp) :: gU, gD, suu, sdd, sud

    ! Exchange energy density per spin
    real(dp) :: ex_den_up, ex_den_dn

    ! Temporary variables
    real(dp) :: rho2_u, rho2_d, kF2_u, kF2_d, s_u, s_d
    real(dp) :: s2_u, s2_d, s2_c
    real(dp) :: zeta, zeta_p, dzeta_dru, dzeta_drd, dzp_dzeta, dx1_dzeta

    if (nSpin /= 1 .and. nSpin /= 2) then
      call die('NNGGAXC: nSpin>2 not implemented')
    end if

    if (.not.nn_weights_loaded) then
      ! Read NN parameter prefix from the input file (fdf) the first time.
      if (len_trim(nn_prefix) == 0) then
        nn_prefix = fdf_string('NNXC.Prefix', 'nn_params/xc_nn')
      end if
      call load_xc_nn_weights(trim(nn_prefix))
      nn_weights_loaded = .true.
    end if

    tiny = 1.0d-14
    pi   = acos(-1.0_dp)
    b_kF = (3.0_dp*pi*pi)**(1.0_dp/3.0_dp)

    ! Spin handling
    if (nSpin == 1) then
      rho = Dens(1)
      ru  = 0.5_dp * rho
      rd  = 0.5_dp * rho
      Gtot(1) = GDens(1,1)
      Gtot(2) = GDens(2,1)
      Gtot(3) = GDens(3,1)
      GradU = 0.5_dp * GDens(:,1)
      GradD = 0.5_dp * GDens(:,1)
    else
      ru  = Dens(1)
      rd  = Dens(2)
      rho = ru + rd
      Gtot(1) = GDens(1,1) + GDens(1,2)
      Gtot(2) = GDens(2,1) + GDens(2,2)
      Gtot(3) = GDens(3,1) + GDens(3,2)
      GradU = GDens(:,1)
      GradD = GDens(:,2)
    end if

    if (rho <= tiny) then
      EX = 0.0_dp; EC = 0.0_dp
      dEXdD(:)    = 0.0_dp
      dECdD(:)    = 0.0_dp
      dEXdGD(:,:) = 0.0_dp
      dECdGD(:,:) = 0.0_dp
      return
    end if

    gU = sqrt(sum(GradU**2))
    gD = sqrt(sum(GradD**2))
    g = sqrt(Gtot(1)**2 + Gtot(2)**2 + Gtot(3)**2)
    suu = sum(GradU**2)
    sdd = sum(GradD**2)
    sud = sum(GradU*GradD)

    densx(1) = max(tiny, 2.0_dp*ru)
    call exchng(IREL, 1, densx, exu_up, vxu_up)

    densx(1) = max(tiny, 2.0_dp*rd)
    call exchng(IREL, 1, densx, exu_dn, vxu_dn)

    ex_den_up = ru * exu_up
    ex_den_dn = rd * exu_dn

    if (nSpin == 1) then
      Dlda(1) = 0.5_dp*rho
      Dlda(2) = 0.5_dp*rho
    else
      Dlda(1) = ru
      Dlda(2) = rd
    end if
    call pw92c(2, Dlda, ecunif, vcunif)
    epsc_LDA = ecunif
    ec_den   = rho * epsc_LDA

    ! Exchange NN (up)
    rho2_u = max(tiny, 2.0_dp*ru)
    kF2_u = (3.0_dp*pi*pi*rho2_u)**(1.0_dp/3.0_dp)
    s_u = gU / (kF2_u * rho2_u + tiny_denom)
    if (s_u < -1.0_dp) s_u = 0.0_dp
    x0_u = log(rho2_u**(1.0_dp/3.0_dp) + 1.0d-5)
    x1_u = log(1.0d-5 + 1.0_dp)
    s2_u = s_u*s_u
    x2_u = (1.0_dp - exp(-s2_u)) * log(1.0_dp + s_u)
    dx2_ds_u = 2.0_dp*s_u*exp(-s2_u)*log(1.0_dp + s_u) + (1.0_dp - exp(-s2_u)) / (1.0_dp + s_u)
    call fx_forward_and_grad3(x0_u, x1_u, x2_u, Fx_u, dFx_dx0_u, dFx_dx1_u, dFx_dx2_u)

    ! Exchange NN (down)
    rho2_d = max(tiny, 2.0_dp*rd)
    kF2_d = (3.0_dp*pi*pi*rho2_d)**(1.0_dp/3.0_dp)
    s_d = gD / (kF2_d * rho2_d + tiny_denom)
    if (s_d < -1.0_dp) s_d = 0.0_dp
    x0_d = log(rho2_d**(1.0_dp/3.0_dp) + 1.0d-5)
    x1_d = log(1.0d-5 + 1.0_dp)
    s2_d = s_d*s_d
    x2_d = (1.0_dp - exp(-s2_d)) * log(1.0_dp + s_d)
    dx2_ds_d = 2.0_dp*s_d*exp(-s2_d)*log(1.0_dp + s_d) + (1.0_dp - exp(-s2_d)) / (1.0_dp + s_d)
    call fx_forward_and_grad3(x0_d, x1_d, x2_d, Fx_d, dFx_dx0_d, dFx_dx1_d, dFx_dx2_d)

    ex_den = ex_den_up*Fx_u + ex_den_dn*Fx_d
    EX = ex_den / rho

    ! Correlation NN (total)
    zeta = (ru - rd) / rho
    zeta = max(-1.0_dp, min(1.0_dp, zeta))
    zeta_p = 0.5_dp * ( (1.0_dp + zeta)**(4.0_dp/3.0_dp) + (1.0_dp - zeta)**(4.0_dp/3.0_dp) )
    x1_c = log(zeta_p + 1.0d-5)
    x0_c = log(rho**(1.0_dp/3.0_dp) + 1.0d-5)
    kF = (3.0_dp*pi*pi*rho)**(1.0_dp/3.0_dp)
    s = g / (2.0_dp * kF * rho + tiny_denom)
    if (s < -1.0_dp) s = 0.0_dp
    s2_c = s*s
    x2_c = (1.0_dp - exp(-s2_c)) * log(1.0_dp + s)
    dx2_ds_c = 2.0_dp*s*exp(-s2_c)*log(1.0_dp + s) + (1.0_dp - exp(-s2_c)) / (1.0_dp + s)
    call fc_forward_and_grad3(x0_c, x1_c, x2_c, Fc, dFc_dx0, dFc_dx1, dFc_dx2)
    EC = Fc * epsc_LDA

    ! Density derivatives
    dx0_drho_u = ( (1.0_dp/3.0_dp) * rho2_u**(-2.0_dp/3.0_dp) ) / (rho2_u**(1.0_dp/3.0_dp) + 1.0d-5)
    ds_drho_u = -(4.0_dp/3.0_dp) * s_u / rho2_u
    ds_dg_u = 1.0_dp / (kF2_u * rho2_u)
    dx2_drho_u = dx2_ds_u * ds_drho_u
    dFx_drho_u = (dFx_dx0_u * dx0_drho_u + dFx_dx2_u * dx2_drho_u) * 2.0_dp
    dedrho_x(1) = Fx_u * vxu_up(1) + ex_den_up * dFx_drho_u

    dx0_drho_d = ( (1.0_dp/3.0_dp) * rho2_d**(-2.0_dp/3.0_dp) ) / (rho2_d**(1.0_dp/3.0_dp) + 1.0d-5)
    ds_drho_d = -(4.0_dp/3.0_dp) * s_d / rho2_d
    ds_dg_d = 1.0_dp / (kF2_d * rho2_d)
    dx2_drho_d = dx2_ds_d * ds_drho_d
    dFx_drho_d = (dFx_dx0_d * dx0_drho_d + dFx_dx2_d * dx2_drho_d) * 2.0_dp
    dedrho_x(2) = Fx_d * vxu_dn(1) + ex_den_dn * dFx_drho_d

    dx0_drho_c = ( (1.0_dp/3.0_dp) * rho**(-2.0_dp/3.0_dp) ) / (rho**(1.0_dp/3.0_dp) + 1.0d-5)
    dzeta_dru = (1.0_dp - zeta) / rho
    dzeta_drd = -(1.0_dp + zeta) / rho
    dzp_dzeta = (2.0_dp/3.0_dp) * ( (1.0_dp + zeta)**(1.0_dp/3.0_dp) - (1.0_dp - zeta)**(1.0_dp/3.0_dp) )
    dx1_dzeta = dzp_dzeta / (zeta_p + 1.0d-5)
    dx1_dru = dx1_dzeta * dzeta_dru
    dx1_drd = dx1_dzeta * dzeta_drd
    ds_drho_c = -(4.0_dp/3.0_dp) * s / rho
    ds_dg_c = 1.0_dp / (2.0_dp * b_kF * rho**(4.0_dp/3.0_dp))
    dx2_drho_c = dx2_ds_c * ds_drho_c
    dx2_dg_c = dx2_ds_c * ds_dg_c
    dedrho_c(1) = Fc * vcunif(1) + ec_den * (dFc_dx0 * dx0_drho_c + dFc_dx1 * dx1_dru + dFc_dx2 * dx2_drho_c)
    dedrho_c(2) = Fc * vcunif(2) + ec_den * (dFc_dx0 * dx0_drho_c + dFc_dx1 * dx1_drd + dFc_dx2 * dx2_drho_c)

    ! Gradient derivatives
    dFx_dg_u = dFx_dx2_u * dx2_ds_u * ds_dg_u
    if (gU > tiny) then
      dedg_x_u = ex_den_up * dFx_dg_u
    else
      dedg_x_u = 0.0_dp
    end if

    dFx_dg_d = dFx_dx2_d * dx2_ds_d * ds_dg_d
    if (gD > tiny) then
      dedg_x_d = ex_den_dn * dFx_dg_d
    else
      dedg_x_d = 0.0_dp
    end if

    dedg_c = ec_den * (dFc_dx2 * dx2_dg_c)

    if (nSpin == 1) then
      if (gU > tiny .and. gD > tiny) then
        do k = 1,3
          dEXdGD(k,1) = 0.5_dp * (dedg_x_u * GradU(k)/gU + dedg_x_d * GradD(k)/gD)
          dECdGD(k,1) = dedg_c * (Gtot(k)/g)
        end do
      else
        dEXdGD(:,1) = 0.0_dp
        dECdGD(:,1) = 0.0_dp
      end if
    else
      if (gU > tiny) then
        do k = 1,3
          dEXdGD(k,1) = dedg_x_u * GradU(k)/gU
          dECdGD(k,1) = dedg_c * (Gtot(k)/g)
        end do
      else
        dEXdGD(:,1) = 0.0_dp
        dECdGD(:,1) = 0.0_dp
      end if
      if (gD > tiny) then
        do k = 1,3
          dEXdGD(k,2) = dedg_x_d * GradD(k)/gD
          dECdGD(k,2) = dedg_c * (Gtot(k)/g)
        end do
      else
        dEXdGD(:,2) = 0.0_dp
        dECdGD(:,2) = 0.0_dp
      end if
    end if

    if (nSpin == 1) then
      dEXdD(1) = 0.5_dp * (dedrho_x(1) + dedrho_x(2))
      dECdD(1) = 0.5_dp * (dedrho_c(1) + dedrho_c(2))
    else
      dEXdD(1) = dedrho_x(1)
      dEXdD(2) = dedrho_x(2)
      dECdD(1) = dedrho_c(1)
      dECdD(2) = dedrho_c(2)
    end if

  end subroutine NNGGAXC

end module xc_nn_siesta