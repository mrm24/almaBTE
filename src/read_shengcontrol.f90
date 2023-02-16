! Copyright 2015-2018 The ALMA Project Developers
!
! Licensed under the Apache License, Version 2.0 (the "License");
! you may not use this file except in compliance with the License.
! You may obtain a copy of the License at
!
!   http:!www.apache.org/licenses/LICENSE-2.0
!
! Unless required by applicable law or agreed to in writing, software
! distributed under the License is distributed on an "AS IS" BASIS,
! WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
! implied. See the License for the specific language governing
! permissions and limitations under the License.

! @file
! Small piece of Fortran code to read settings from a namelist-format CONTROL

module read_shengcontrol
  use iso_fortran_env
  use iso_c_binding
  implicit none

  ! Data structure containing the information in an allocations namelist
  type, bind(C) :: sheng_allocations
     ! Number of elements in the compound
     integer(kind=c_int) :: nelements
     ! Number of atoms in the compound
     integer(kind=c_int) :: natoms
     ! Number of divisions along each direction of the q-point grid
     integer(kind=c_int) :: ngrid(3)
     ! Number of orientations for nanowires (not supported)
     integer(kind=c_int) :: norientations
  end type sheng_allocations
  ! Data structure containing the information in a parameters namelist
  type, bind(C) :: sheng_parameters
     ! Temperature for single-T calculations
     real(kind=c_double) :: T
     ! Broadening factor for the Gaussians
     real(kind=c_double) :: scalebroad
     ! Minimum radius for nanowires (not supported)
     real(kind=c_double) :: rmin
     ! Maximum radius for nanowires (not supported)
     real(kind=c_double) :: rmax
     ! Radius step for nanowires (not supported)
     real(kind=c_double) :: dr
     ! Maximum number of iterations of the iterative method
     integer(kind=c_int) :: maxiter
     ! Number of divisions for the DOS and cumulative kappa calculations.
     integer(kind=c_int) :: nticks
     ! Tolerance criterion for the iterative solver
     real(kind=c_double) :: eps
     ! Minimum temperature for T-sweep calculations
     real(kind=c_double) :: T_min
     ! Maximum temperature for T-sweep calculations
     real(kind=c_double) :: T_max
     ! Temperature step for T-sweep calculations
     real(kind=c_double) :: T_step
     ! Optional angular frequency cutoff for anharmonic calculations (not supported)
     real(kind=c_double) :: omega_max
  end type sheng_parameters
  ! Data structure containing the information in a flags namelist
  type, bind(C) :: sheng_flags
     ! Include a nonanalytic correction?
     logical(kind=c_bool) :: nonanalytic
     ! Run the self-consistent solver or stay at the RTA level?
     logical(kind=c_bool) :: convergence
     ! Include isotopic scattering?
     logical(kind=c_bool) :: isotopes
     ! Fill in the masses and g factors automatically?
     logical(kind=c_bool) :: autoisotopes
     ! Compute the thermal conductivity of nanowires (not supported)
     logical(kind=c_bool) :: nanowires
     ! Compute only harmonic quantities
     logical(kind=c_bool) :: onlyharmonic
     ! Read the inputs in ESPRESSO format (not supported)
     logical(kind=c_bool) :: espresso
  end type sheng_flags
contains
  ! Read the allocations namelist from a CONTROL file in the current
  ! directory
  function read_sheng_allocations() bind(C, name="read_sheng_allocations")
    use iso_fortran_env
    use iso_c_binding
    implicit none

    type(sheng_allocations) :: read_sheng_allocations
    integer(kind=c_int) :: nelements
    integer(kind=c_int) :: natoms
    integer(kind=c_int) :: ngrid(3)
    integer(kind=c_int) :: norientations
    namelist /allocations/ nelements, natoms, ngrid, norientations

    nelements = 0
    natoms = 0
    ngrid = 0
    norientations = 0
    open(1, file="CONTROL", status="old")
    read(1, nml=allocations)
    close(1)
    read_sheng_allocations%nelements = nelements
    read_sheng_allocations%natoms = natoms
    read_sheng_allocations%ngrid = ngrid
    read_sheng_allocations%norientations = norientations
  end function read_sheng_allocations
  function read_sheng_parameters() bind(C, name="read_sheng_parameters")
    use iso_fortran_env
    use iso_c_binding
    implicit none

    type(sheng_parameters) :: read_sheng_parameters
    real(kind=c_double) :: T
    real(kind=c_double) :: scalebroad
    real(kind=c_double) :: rmin
    real(kind=c_double) :: rmax
    real(kind=c_double) :: dr
    integer(kind=c_int) :: maxiter
    integer(kind=c_int) :: nticks
    real(kind=c_double) :: eps
    real(kind=c_double) :: T_min
    real(kind=c_double) :: T_max
    real(kind=c_double) :: T_step
    real(kind=c_double) :: omega_max
    namelist /parameters/ T, scalebroad, rmin, rmax, dr, maxiter, nticks, eps, &
         T_min ,T_max, T_step, omega_max

    T = 0
    T_min = 0
    scalebroad = 1.0
    rmin = 5.0
    rmax = 505.0
    dr = 100.0
    maxiter = 1000
    nticks = 100
    eps = 1e-5
    omega_max = -1.
    open(1, file="CONTROL", status="old")
    read(1, nml=parameters)
    close(1)
    read_sheng_parameters%T = T
    read_sheng_parameters%scalebroad = scalebroad
    read_sheng_parameters%rmin = rmin
    read_sheng_parameters%rmax = rmax
    read_sheng_parameters%dr = dr
    read_sheng_parameters%maxiter = maxiter
    read_sheng_parameters%nticks = nticks
    read_sheng_parameters%eps = eps
    read_sheng_parameters%T_min = T_min
    read_sheng_parameters%T_max = T_max
    read_sheng_parameters%T_step = T_step
    read_sheng_parameters%omega_max = omega_max
  end function read_sheng_parameters
  ! Read the flags namelist from a CONTROL file in the current directory
  function read_sheng_flags() bind(C, name="read_sheng_flags")
    use iso_fortran_env
    use iso_c_binding
    implicit none

    type(sheng_flags) :: read_sheng_flags
    logical(kind=c_bool) :: nonanalytic
    logical(kind=c_bool) :: convergence
    logical(kind=c_bool) :: isotopes
    logical(kind=c_bool) :: autoisotopes
    logical(kind=c_bool) :: nanowires
    logical(kind=c_bool) :: onlyharmonic
    logical(kind=c_bool) :: espresso
    namelist /flags/ nonanalytic, convergence, isotopes, autoisotopes, &
         nanowires, onlyharmonic, espresso
    
    nonanalytic=.true.
    convergence=.true.
    isotopes=.true.
    autoisotopes=.true.
    nanowires=.false.
    onlyharmonic=.false.
    espresso=.false.
    open(1, file="CONTROL", status="old")
    read(1, nml=flags)
    close(1)
    read_sheng_flags%nonanalytic = nonanalytic
    read_sheng_flags%convergence = convergence
    read_sheng_flags%isotopes = isotopes
    read_sheng_flags%autoisotopes = autoisotopes
    read_sheng_flags%nanowires = nanowires
    read_sheng_flags%onlyharmonic = onlyharmonic
    read_sheng_flags%espresso = espresso
  end function read_sheng_flags
  ! Read the supercell size from a CONTROL file in the current directory.
  ! This is only used in connection with the harmonic IFCs.
  subroutine read_sheng_scell(allocs, na, nb, nc) &
    bind(C, name="read_sheng_scell")
    use iso_fortran_env
    use iso_c_binding
    implicit none

    type(sheng_allocations) :: allocs
    integer(kind=c_int) :: na
    integer(kind=c_int) :: nb
    integer(kind=c_int) :: nc
    real(kind=c_double) :: lfactor
    real(kind=c_double) :: lattvec(3, 3)
    character(len=3) :: elements(allocs%nelements)
    integer(kind=c_int) :: types(allocs%natoms)
    integer(kind=c_int), allocatable :: orientations(:, :)
    integer(kind=c_int) :: scell(3)
    real(kind=c_double) :: positions(3, allocs%natoms)
    real(kind=c_double) :: masses(allocs%nelements)
    real(kind=c_double) :: gfactors(allocs%natoms)
    real(kind=c_double) :: epsilon(3, 3)
    real(kind=c_double) :: born(3, 3, allocs%natoms)
    namelist /crystal/ lfactor, lattvec, elements, types, positions, &
      masses, gfactors, epsilon, born, scell, orientations

    if (allocs%norientations .ne. 0) then
      allocate(orientations(3, allocs%norientations))
    end if
    scell = -1
    open(1, file="CONTROL", status="old")
    read(1, nml=crystal)
    close(1)
    na = scell(1)
    nb = scell(2)
    nc = scell(3)
    if (allocs%norientations .ne. 0) then
      deallocate(orientations)
    end if
  end subroutine read_sheng_scell
  ! Read the lattice vectors from a CONTROL file in the current directory.
  ! The returned vectors include the lfactor factor.
  subroutine read_sheng_lattvec(allocs, lattvec) &
    bind(C, name="read_sheng_lattvec")
    use iso_fortran_env
    use iso_c_binding
    implicit none

    type(sheng_allocations) :: allocs
    real(kind=c_double) :: lattvec(3, 3)
    real(kind=c_double) :: lfactor
    real(kind=c_double) :: epsilon(3, 3)
    character(len=3) :: elements(allocs%nelements)
    integer(kind=c_int) :: types(allocs%natoms)
    integer(kind=c_int), allocatable :: orientations(:, :)
    integer(kind=c_int) :: scell(3)
    real(kind=c_double) :: positions(3, allocs%natoms)
    real(kind=c_double) :: masses(allocs%nelements)
    real(kind=c_double) :: gfactors(allocs%natoms)
    real(kind=c_double) :: born(3, 3, allocs%natoms)
    namelist /crystal/ lfactor, lattvec, elements, types, positions, &
      masses, gfactors, epsilon, born, scell, orientations

    if (allocs%norientations .ne. 0) then
      allocate(orientations(3, allocs%norientations))
    end if
    lfactor = 1.
    open(1, file="CONTROL", status="old")
    read(1, nml=crystal)
    close(1)
    lattvec = lfactor * lattvec
    lattvec = transpose(lattvec)
    if (allocs%norientations .ne. 0) then
      deallocate(orientations)
    end if
  end subroutine read_sheng_lattvec
  ! Read the dielectric tensor from a CONTROL file in the current directory.
  subroutine read_sheng_epsilon(allocs, epsilon) &
    bind(C, name="read_sheng_epsilon")
    use iso_fortran_env
    use iso_c_binding
    implicit none

    type(sheng_allocations) :: allocs
    real(kind=c_double) :: epsilon(3, 3)
    real(kind=c_double) :: lfactor
    real(kind=c_double) :: lattvec(3, 3)
    character(len=3) :: elements(allocs%nelements)
    integer(kind=c_int) :: types(allocs%natoms)
    integer(kind=c_int), allocatable :: orientations(:, :)
    integer(kind=c_int) :: scell(3)
    real(kind=c_double) :: positions(3, allocs%natoms)
    real(kind=c_double) :: masses(allocs%nelements)
    real(kind=c_double) :: gfactors(allocs%natoms)
    real(kind=c_double) :: born(3, 3, allocs%natoms)
    namelist /crystal/ lfactor, lattvec, elements, types, positions, &
      masses, gfactors, epsilon, born, scell, orientations

    if (allocs%norientations .ne. 0) then
      allocate(orientations(3, allocs%norientations))
    end if
    epsilon = 1.
    open(1, file="CONTROL", status="old")
    read(1, nml=crystal)
    close(1)
    epsilon = transpose(epsilon)
    if (allocs%norientations .ne. 0) then
      deallocate(orientations)
    end if
  end subroutine read_sheng_epsilon
  ! Read the list of atom types from a CONTROL file in the current
  ! directory.
  subroutine read_sheng_types(allocs, types) &
    bind(C, name="read_sheng_types")
    use iso_fortran_env
    use iso_c_binding
    implicit none

    type(sheng_allocations) :: allocs
    integer(kind=c_int) :: types(allocs%natoms)
    integer(kind=c_int) :: na
    integer(kind=c_int) :: nb
    integer(kind=c_int) :: nc
    real(kind=c_double) :: lfactor
    real(kind=c_double) :: lattvec(3, 3)
    character(len=3) :: elements(allocs%nelements)
    integer(kind=c_int), allocatable :: orientations(:, :)
    integer(kind=c_int) :: scell(3)
    real(kind=c_double) :: positions(3, allocs%natoms)
    real(kind=c_double) :: masses(allocs%nelements)
    real(kind=c_double) :: gfactors(allocs%natoms)
    real(kind=c_double) :: epsilon(3, 3)
    real(kind=c_double) :: born(3, 3, allocs%natoms)
    namelist /crystal/ lfactor, lattvec, elements, types, positions, &
      masses, gfactors, epsilon, born, scell, orientations

    if (allocs%norientations .ne. 0) then
      allocate(orientations(3, allocs%norientations))
    end if
    types = -1
    open(1, file="CONTROL", status="old")
    read(1, nml=crystal)
    close(1)
    types = types - 1
    if (allocs%norientations .ne. 0) then
      deallocate(orientations)
    end if
  end subroutine read_sheng_types
  ! Read the list of element masses from a CONTROL file in the current
  ! directory.
  subroutine read_sheng_masses(allocs, masses) &
    bind(C, name="read_sheng_masses")
    use iso_fortran_env
    use iso_c_binding
    implicit none

    type(sheng_allocations) :: allocs
    real(kind=c_double) :: masses(allocs%nelements)
    integer(kind=c_int) :: na
    integer(kind=c_int) :: nb
    integer(kind=c_int) :: nc
    real(kind=c_double) :: lfactor
    real(kind=c_double) :: lattvec(3, 3)
    character(len=3) :: elements(allocs%nelements)
    integer(kind=c_int) :: types(allocs%natoms)
    integer(kind=c_int), allocatable :: orientations(:, :)
    integer(kind=c_int) :: scell(3)
    real(kind=c_double) :: positions(3, allocs%natoms)
    real(kind=c_double) :: gfactors(allocs%natoms)
    real(kind=c_double) :: epsilon(3, 3)
    real(kind=c_double) :: born(3, 3, allocs%natoms)
    namelist /crystal/ lfactor, lattvec, elements, types, positions, &
      masses, gfactors, epsilon, born, scell, orientations

    if (allocs%norientations .ne. 0) then
      allocate(orientations(3, allocs%norientations))
    end if
    masses = -1.
    open(1, file="CONTROL", status="old")
    read(1, nml=crystal)
    close(1)
    if (allocs%norientations .ne. 0) then
      deallocate(orientations)
    end if
  end subroutine read_sheng_masses
  ! Read the list of of mass disorder "g" factors from a CONTROL file
  ! in the current directory.
  subroutine read_sheng_gfactors(allocs, gfactors) &
    bind(C, name="read_sheng_gfactors")
    use iso_fortran_env
    use iso_c_binding
    implicit none

    type(sheng_allocations) :: allocs
    real(kind=c_double) :: gfactors(allocs%natoms)
    integer(kind=c_int) :: na
    integer(kind=c_int) :: nb
    integer(kind=c_int) :: nc
    real(kind=c_double) :: lfactor
    real(kind=c_double) :: lattvec(3, 3)
    character(len=3) :: elements(allocs%nelements)
    integer(kind=c_int) :: types(allocs%natoms)
    integer(kind=c_int), allocatable :: orientations(:, :)
    integer(kind=c_int) :: scell(3)
    real(kind=c_double) :: positions(3, allocs%natoms)
    real(kind=c_double) :: masses(allocs%nelements)
    real(kind=c_double) :: epsilon(3, 3)
    real(kind=c_double) :: born(3, 3, allocs%natoms)
    namelist /crystal/ lfactor, lattvec, elements, types, positions, &
      masses, gfactors, epsilon, born, scell, orientations

    if (allocs%norientations .ne. 0) then
      allocate(orientations(3, allocs%norientations))
    end if
    gfactors = -1.
    open(1, file="CONTROL", status="old")
    read(1, nml=crystal)
    close(1)
    if (allocs%norientations .ne. 0) then
      deallocate(orientations)
    end if
  end subroutine read_sheng_gfactors
  ! Read the coordinates of an atom from a CONTROL file in the current
  ! directory.
  subroutine read_sheng_position(allocs, index, position) &
    bind(C, name="read_sheng_position")
    use iso_fortran_env
    use iso_c_binding
    implicit none

    type(sheng_allocations) :: allocs
    integer(kind=c_int), value :: index
    real(kind=c_double) :: position(3)
    real(kind=c_double) :: lfactor
    real(kind=c_double) :: lattvec(3, 3)
    character(len=3) :: elements(allocs%nelements)
    integer(kind=c_int) :: types(allocs%natoms)
    integer(kind=c_int), allocatable :: orientations(:, :)
    integer(kind=c_int) :: scell(3)
    real(kind=c_double) :: positions(3, allocs%natoms)
    real(kind=c_double) :: masses(allocs%nelements)
    real(kind=c_double) :: gfactors(allocs%natoms)
    real(kind=c_double) :: epsilon(3, 3)
    real(kind=c_double) :: born(3, 3, allocs%natoms)
    namelist /crystal/ lfactor, lattvec, elements, types, positions, &
      masses, gfactors, epsilon, born, scell, orientations

    if (allocs%norientations .ne. 0) then
      allocate(orientations(3, allocs%norientations))
    end if
    open(1, file="CONTROL", status="old")
    read(1, nml=crystal)
    close(1)
    position = positions(:, index + 1)
    if (allocs%norientations .ne. 0) then
      deallocate(orientations)
    end if
  end subroutine read_sheng_position
  ! Read the Born effective charge tensor of an atom from a CONTROL file
  ! in the current directory.
  subroutine read_sheng_born(allocs, index, cborn) &
    bind(C, name="read_sheng_born")
    use iso_fortran_env
    use iso_c_binding
    implicit none

    type(sheng_allocations) :: allocs
    integer(kind=c_int), value :: index
    real(kind=c_double) :: cborn(3, 3)
    real(kind=c_double) :: lfactor
    real(kind=c_double) :: lattvec(3, 3)
    character(len=3) :: elements(allocs%nelements)
    integer(kind=c_int) :: types(allocs%natoms)
    integer(kind=c_int), allocatable :: orientations(:, :)
    integer(kind=c_int) :: scell(3)
    real(kind=c_double) :: positions(3, allocs%natoms)
    real(kind=c_double) :: masses(allocs%nelements)
    real(kind=c_double) :: gfactors(allocs%natoms)
    real(kind=c_double) :: epsilon(3, 3)
    real(kind=c_double) :: born(3, 3, allocs%natoms)
    namelist /crystal/ lfactor, lattvec, elements, types, positions, &
      masses, gfactors, epsilon, born, scell, orientations

    if (allocs%norientations .ne. 0) then
      allocate(orientations(3, allocs%norientations))
    end if
    born = 0.
    open(1, file="CONTROL", status="old")
    read(1, nml=crystal)
    close(1)
    cborn = transpose(born(:, :, index + 1))
    if (allocs%norientations .ne. 0) then
      deallocate(orientations)
    end if
  end subroutine read_sheng_born
  ! Read the Born effective charge tensor of an atom from a CONTROL file
  ! in the current directory.
  subroutine read_sheng_element(allocs, index, element) &
    bind(C, name="read_sheng_element")
    use iso_fortran_env
    use iso_c_binding
    implicit none

    type(sheng_allocations) :: allocs
    integer(kind=c_int), value :: index
    character(kind=c_char, len=1) :: element(4)
    real(kind=c_double) :: lfactor
    real(kind=c_double) :: lattvec(3, 3)
    character(len=3) :: elements(allocs%nelements)
    integer(kind=c_int) :: types(allocs%natoms)
    integer(kind=c_int), allocatable :: orientations(:, :)
    integer(kind=c_int) :: scell(3)
    real(kind=c_double) :: positions(3, allocs%natoms)
    real(kind=c_double) :: masses(allocs%nelements)
    real(kind=c_double) :: gfactors(allocs%natoms)
    real(kind=c_double) :: epsilon(3, 3)
    real(kind=c_double) :: born(3, 3, allocs%natoms)
    namelist /crystal/ lfactor, lattvec, elements, types, positions, &
      masses, gfactors, epsilon, born, scell, orientations

    integer(kind=c_int) :: i

    if (allocs%norientations .ne. 0) then
      allocate(orientations(3, allocs%norientations))
    end if
    do i=1, allocs%natoms
      elements(i) = ""
    end do
    open(1, file="CONTROL", status="old")
    read(1, nml=crystal)
    close(1)
    element(1:3) = transfer(elements(index + 1), " ", size=3)
    element(4) = C_NULL_CHAR
    if (allocs%norientations .ne. 0) then
      deallocate(orientations)
    end if
  end subroutine read_sheng_element
end module read_shengcontrol
