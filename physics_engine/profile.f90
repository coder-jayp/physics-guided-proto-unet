module disk_physics
    use iso_c_binding
    implicit none
contains
    subroutine calculate_radial_profile(image, nx_ptr, ny_ptr, profile, n_bins_ptr) bind(c, name='calculate_radial_profile')
        ! Removed the 'value' attribute to match Python's byref (pointers)
        integer(c_int), intent(in) :: nx_ptr, ny_ptr, n_bins_ptr
        real(c_float), intent(in) :: image(*)  ! Use flat array for safer memory mapping
        real(c_float), intent(inout) :: profile(*)
        
        integer :: i, j, bin_idx, nx, ny, n_bins
        real(c_float) :: dx, dy, dist, max_dist
        integer, allocatable :: counts(:)

        ! Dereference the pointers for easier use
        nx = nx_ptr
        ny = ny_ptr
        n_bins = n_bins_ptr

        allocate(counts(n_bins))
        counts = 0
        
        ! Initialize profile (important if not done in Python)
        do i = 1, n_bins
            profile(i) = 0.0
        end do

        max_dist = sqrt(real(nx/2, c_float)**2 + real(ny/2, c_float)**2)

        !$omp parallel do private(i, j, dx, dy, dist, bin_idx) shared(image, profile, counts)
        do j = 1, ny
            do i = 1, nx
                dx = real(i - nx/2, c_float)
                dy = real(j - ny/2, c_float)
                dist = sqrt(dx**2 + dy**2)
                
                ! Use min/max to strictly enforce array boundaries
                bin_idx = int((dist / max_dist) * (n_bins - 1)) + 1
                
                if (bin_idx >= 1 .and. bin_idx <= n_bins) then
                    !$omp atomic
                    profile(bin_idx) = profile(bin_idx) + image(i + (j-1)*nx)
                    !$omp atomic
                    counts(bin_idx) = counts(bin_idx) + 1
                end if
            end do
        end do

        ! Average the sums
        do i = 1, n_bins
            if (counts(i) > 0) then
                profile(i) = profile(i) / real(counts(i), c_float)
            end if
        end do
        
        deallocate(counts)
    end subroutine calculate_radial_profile
end module disk_physics