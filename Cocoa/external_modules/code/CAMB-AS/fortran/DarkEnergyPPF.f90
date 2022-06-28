    module DarkEnergyPPF
    use DarkEnergyInterface
    use classes
    use results !KZ-edit
    implicit none

    private

    type, extends(TDarkEnergyEqnOfState) :: TDarkEnergyPPF
        !integer :: shear_model = 1.0 !KZ-edit: model of Anisotropic stress
        !real(dl) :: g0_ppf = 0.0 !KZ-edit: ppf parameter g0
        !real(dl) :: c_Gamma_ppf = 1.0 !KZ-edit
        !real(dl) :: omT_over_ome !KZ-edit: add parameter Omega_T/Omega_DE
    contains
    procedure :: ReadParams => TDarkEnergyPPF_ReadParams
    procedure, nopass :: PythonClass => TDarkEnergyPPF_PythonClass
    procedure :: Init => TDarkEnergyPPF_Init
    procedure :: PerturbedStressEnergy => TDarkEnergyPPF_PerturbedStressEnergy
    procedure :: diff_rhopi_Add_Term => TDarkEnergyPPF_diff_rhopi_Add_Term
    procedure, nopass :: SelfPointer => TDarkEnergyPPF_SelfPointer
    procedure, private :: setcgammappf
    end type TDarkEnergyPPF

    public TDarkEnergyPPF
    contains

    subroutine TDarkEnergyPPF_ReadParams(this, Ini)
    use IniObjects
    class(TDarkEnergyPPF) :: this
    class(TIniFile), intent(in) :: Ini

    call this%TDarkEnergyEqnOfState%ReadParams(Ini)
    this%shear_model = Ini%Read_Double('shear_model', 0.d0) !KZ-edit
    this%cs2_lam = Ini%Read_Double('cs2_lam', 1.d0)
    this%g0_ppf = Ini%Read_Double('g0_ppf', 1.d0) !KZ-edit: read g0 from ini file
    this%c_Gamma_ppf = Ini%Read_Double('c_Gamma_ppf', 1.d0) !KZ-edit: read c_gamma from ini file 
    this%c_g_ppf = Ini%Read_Double('c_g_ppf', 0.01_dl) !KZ-edit: read c_g from ini file
    this%c_gamma_k_H_square_max = Ini%Read_Double('c_gamma_k_H_square_max', 30.0_dl) !KZ-edit: read c_g from ini file  
    if (this%cs2_lam /= 1.d0) error stop 'cs2_lam not supported by PPF model'
    !call this%setcgammappf

    end subroutine TDarkEnergyPPF_ReadParams

    function TDarkEnergyPPF_PythonClass()
    character(LEN=:), allocatable :: TDarkEnergyPPF_PythonClass

    TDarkEnergyPPF_PythonClass = 'DarkEnergyPPF'
    end function TDarkEnergyPPF_PythonClass


    subroutine TDarkEnergyPPF_SelfPointer(cptr,P)
    use iso_c_binding
    Type(c_ptr) :: cptr
    Type (TDarkEnergyPPF), pointer :: PType
    class (TPythonInterfacedClass), pointer :: P

    call c_f_pointer(cptr, PType)
    P => PType

    end subroutine TDarkEnergyPPF_SelfPointer

    subroutine TDarkEnergyPPF_Init(this, State)
    use classes
    class(TDarkEnergyPPF), intent(inout) :: this
    class(TCAMBdata), intent(in), target :: State

    !KZ-edit begin
    !select type(State)
    !class is (CAMBdata)
    !    this%omT_over_ome = (1 - State%Omega_de) / State%Omega_de !parameter in g_sh
    !end select

    !this%is_cosmological_constant = .FALSE.
    !print *, "Is using PPF" !for debug
    !print *, this%g0 !for debug

    !KZ-edit end

    call this%TDarkEnergyEqnOfState%Init(State)
    if (this%is_cosmological_constant) then
        this%num_perturb_equations = 0
    else
        this%num_perturb_equations = 1
    end if


    end subroutine TDarkEnergyPPF_Init

    subroutine setcgammappf(this)
    class(TDarkEnergyPPF) :: this

    !this%c_Gamma_ppf = 0.4_dl * sqrt(this%cs2_lam) !KZ-edit

    end subroutine setcgammappf


    function TDarkEnergyPPF_diff_rhopi_Add_Term(this, dgrhoe, dgqe, grho, gpres, w,  grhok, adotoa, &
        Kf1, k, grhov_t, z, k2, yprime, y, w_ix, dgpie) result(ppiedot)
    !Get derivative of anisotropic stress
    class(TDarkEnergyPPF), intent(in) :: this
    real(dl), intent(in) :: dgrhoe, dgqe, grho, gpres, w, grhok, adotoa, &
        k, grhov_t, z, k2, yprime(:), y(:), Kf1
    integer, intent(in) :: w_ix
    real(dl), intent(in) :: dgpie
    real(dl) :: ppiedot, hdotoh

    if (this%is_cosmological_constant .or. this%no_perturbations) then
        ppiedot = 0
    else
        hdotoh = (-3._dl * grho - 3._dl * gpres - 2._dl * grhok) / 6._dl / adotoa
        ppiedot = 3._dl * dgrhoe + dgqe * &
            (12._dl / k * adotoa + k / adotoa - 3._dl / k * (adotoa + hdotoh)) + &
            grhov_t * (1 + w) * k * z / adotoa - 2._dl * k2 * Kf1 * &
            (yprime(w_ix) / adotoa - 2._dl * y(w_ix)) + 2._dl*Kf1*dgpie !KZ-edit: add Anisotropic stress from DE
        ppiedot = ppiedot * adotoa / Kf1
    end if

    end function TDarkEnergyPPF_diff_rhopi_Add_Term


    subroutine TDarkEnergyPPF_PerturbedStressEnergy(this, dgrhoe, dgqe, dgpie, sigma_ppf, & ! KZ-edit: add dgpie 
        a, dgq, dgrho, grho, grhov_t, w, gpres_noDE, etak, adotoa, k, kf1, ay, ayprime, w_ix, &
        grhor_t, grhog_t, pir, pig, pirdot, pigdot, grhonu_t, gpnu_t, dgpi_nu_sum, ppi_nu_dot_sum) !KZ-edit
    class(TDarkEnergyPPF), intent(inout) :: this
    real(dl), intent(out) :: dgrhoe, dgqe, dgpie, sigma_ppf !KZ-edit: add dgpie
    real(dl), intent(in) ::  a, dgq, dgrho, grho, grhov_t, w, gpres_noDE, etak, adotoa, k, kf1
    real(dl), intent(in) :: ay(*)
    real(dl), intent(inout) :: ayprime(*)
    integer, intent(in) :: w_ix
    real(dl), intent(in) :: grhor_t, grhog_t, pir, pig, pirdot, pigdot, grhonu_t, gpnu_t, dgpi_nu_sum, ppi_nu_dot_sum
    real(dl) :: Gamma, S_Gamma, ckH, Gammadot, Fa, sigma
    real(dl) :: grhoT, k2, VT_sync, VT_comoving, dgrho_T_comoving
    real(dl) :: gppf,gdotppf,fGppf,fzetappf,gammacond, g0_ppf, gsh, dgqe_comoving, dgPI_comoving, dgrhoe_comoving    !KZ-edit
    real(dl) :: Phi_minus, dgqT, dgrhoT, dgpiT, ppiTdot    !KZ-edit
    real(dl) :: cg, z0, dz, z, pp, qq, y, dgppf, adotdotoa,  k_H_dot, k_H, gprimeppf, ppiT_prime  !KZ-edit

    if (this%no_perturbations) then
        dgrhoe=0
        dgqe=0
        return
    end if

    !KZ-edit begins: the equations are from https://arxiv.org/abs/0708.1190v2
    k2=k**2
    !ppf

    grhoT = grho - grhov_t
    VT_sync = dgq / (grhoT + gpres_noDE)
    Gamma = ay(w_ix)

    dgqT = dgq !rename
    dgrhoT = dgrho !rename
    ckH = this%c_Gamma_ppf * k / adotoa 
    z = 1/a -1
            
    !dgpiT=grhor_t*pir+grhog_t*pig +grhonu_t * pi_nu_sum
    !ppiT_prime = grhog_t*(pigdot/adotoa-4*pig) + grhor_t*(pirdot/adotoa-4*pir) +grhonu_t*pidot_nu_sum/adotoa-3._dl*(grhonu_t+gpnu_t)*pi_nu_sum
    
    dgpiT=grhor_t*pir+grhog_t*pig + dgpi_nu_sum
    ppiT_prime = grhog_t*(pigdot/adotoa-4*pig) + grhor_t*(pirdot/adotoa-4*pir) + ppi_nu_dot_sum/ adotoa

    !!KZ: for test
    dgpiT = 0.0
    ppiT_prime = 0.0

    ! Model loop begins
    ! Models for Effective Anisotropic Stress in PPF formalism,
    ! For each model, we need to define gppf, c_gamma, and whether it source the anisotropic stress from other component
    ! Notation: dgpiT: p_T*\pi_T in the paper
    !CASE 1:
    if(this%shear_model.eq.1 )then  

        ! prime is derivative wrt to ln(a), dot is wrt to conformal time
        fGppf = 0._dl
        cg = this%c_g_ppf

        adotdotoa = (adotoa*adotoa - gpres_noDE-grhov_t*w)/2.0 !a dotdot over a
        k_H = k/adotoa
        y = (cg*k_H)**2

        gppf= this%g0_ppf*a**(1.5)/(1+(cg*k/adotoa)**2) !KZ
        pp = y/(1.0 + y)
        qq = (1.0 - (gpres_noDE + grhov_t*w)/adotoa**2)/2.0
        gprimeppf = gppf*(1.5 - 2.0*(1.0 - qq)*pp*a*adotoa) 

        fzetappf = 0.4*this%g0_ppf*a**(1.5)

        !print *, "model = ", this%shear_model
        !print *, "g0_ppf = ", this%g0_ppf
        !print *, "c_Gamma_ppf = ", this%c_Gamma_ppf
        !CALL EXIT(1)

    else
        print *, "model = ", this%shear_model
        print *, "g0_ppf = ", this%g0_ppf
        print *, "c_Gamma_ppf = ", this%c_Gamma_ppf
        print *, "c_g_ppf = ", this%c_g_ppf
        print *, "No such Anisotropic model"
        CALL EXIT(1)
    endif

    !!model loop end

    ! EQUATION A3
    dgrho_T_comoving=dgrhoT+3.0_dl*dgq*(adotoa/k)
    ! EQUATION 30 
    Phi_minus = -Gamma + 0.5_dl*dgrho_T_comoving/k2 + 0.5_dl*dgpiT/k2 !Ck = 1 here
    ! EQUATION A7 !The transformation is written as sigma in old camb
    if(etak+1.0 .gt. etak) then
        sigma = (etak + (dgrho + 3.0_dl * adotoa / k * dgq) / 2._dl / k) / kf1 - k * Gamma + k*gppf*Phi_minus
        sigma = sigma / adotoa
    else 
        !print *, "testing etak", etak
        sigma = 0.0_dl
    endif

    VT_comoving = VT_sync + sigma  

    !KZ-edit: source term, and Gamma equation. In comoving gauge
    
    gammacond=abs(Gamma/Phi_minus - fGppf)

    !KZ: Approximation scheme: S_Gamma here as in the public camb, WH put it inside the else argument below
    !KZ: Also under this approximation, there's no S_Gamma=0 in the if argument
    S_Gamma=gppf*(dgpiT + ppiT_prime)
    S_Gamma=S_Gamma-((gppf+fzetappf+gppf*fzetappf)*(grhoT+gpres_noDE) -grhov_t*(1_dl+w))*(VT_comoving)*k/adotoa
    S_Gamma=S_Gamma/2._dl/k2/(1.0_dl + gppf)+ (gprimeppf - 2_dl*gppf)*Phi_minus/(1.0_dl + gppf)
    
    if (ckH*ckH.gt.this%c_gamma_k_H_square_max) then ! !KZ: from class notes: The equation is too stiff for Runge-Kutta when c_gamma_k_H_square is large. Use the asymptotic solution Gamma=Gamma'=0 in that case.
        Gamma=fGppf*Phi_minus
        Gammadot=0._dl
        !S_Gamma = 0.0_dl
    else
        !S_Gamma=gppf*(dgpiT + ppiT_prime)
        !S_Gamma=S_Gamma-((gppf+fzetappf+gppf*fzetappf)*(grhoT+gpres_noDE) -grhov_t*(1_dl+w))*(VT_comoving)*k/adotoa
        !S_Gamma=S_Gamma/2._dl/k2/(1.0_dl + gppf)+ (gprimeppf - 2_dl*gppf)*Phi_minus/(1.0_dl + gppf)

        Gammadot = S_Gamma / (1 + ckH * ckH)- Gamma -ckH * ckH * (Gamma - fGppf * Phi_minus)
        Gammadot = Gammadot * adotoa
    endif
    ayprime(w_ix) = Gammadot !Set this here, and don't use PerturbationEvolve

    ! KZ: get perturbations of DE fluid from Gamma, the perturbations is in synchrnous gauge

    !! EQUATION 40
    Fa = 1.0_dl+3.0_dl*(1.0_dl+gppf)*(grhoT+gpres_noDE)/2.0_dl/k2    
    dgqe_comoving =  S_Gamma - (Gammadot/adotoa) - Gamma + fzetappf*(grhoT + gpres_noDE)*( VT_comoving) / 2._dl / k / adotoa
    dgqe_comoving = -dgqe_comoving*2.0_dl*k*adotoa*(1.0_dl + gppf)/Fa
    dgqe_comoving =  dgqe_comoving + VT_comoving*grhov_t*(1.0_dl + w)    
    !! EQUATION 33
    dgPI_comoving = -2.0_dl*gppf*Phi_minus*k2 
    !! EQUATION 31
    dgrhoe_comoving = -(3.0_dl*adotoa/k)*(dgqe_comoving - VT_comoving*grhov_t*(1.0_dl+w))
    dgrhoe_comoving =  dgrhoe_comoving - dgPI_comoving - 2.0_dl*k2*Gamma   
    !! EQUATION A8
    dgrhoe = dgrhoe_comoving - 3.0_dl*grhov_t*(1.0_dl+w)*VT_sync*adotoa/k
    !! EQUATION A9
    dgqe = dgqe_comoving - grhov_t*(1.0_dl + w)*(VT_comoving - VT_sync)  
    dgpie = -2._dl*gppf*Phi_minus*k2 

    sigma_ppf = sigma


    !KZ-edit end

    end subroutine TDarkEnergyPPF_PerturbedStressEnergy



    end module DarkEnergyPPF
