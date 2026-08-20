! plot3d_read_fortran.f90
!
! A minimal compiled Fortran PLOT3D multi-block grid reader, used as the
! "compiled Fortran" baseline in the lptlib I/O benchmark. It reads the same
! little-endian binary file that lptlib.io.GridIO.read_grid consumes:
!
!     int32                    nb
!     int32 * (3*nb)           ni, nj, nk per block, interleaved
!     float32                  block data, each block (ni,nj,nk,3) in
!                              Fortran (column-major) order
!
! Usage:
!     plot3d_read_fortran <gridfile> <nreps>
!
! The program performs the same task as the Python readers in this benchmark:
! read the file and reconstruct the per-block coordinate arrays in memory. It
! times only that read-and-reconstruct work (system_clock), printing one
! "rep <n> <seconds>" line per repetition. A single untimed pass afterward
! prints "checksum <value>" (float64 sum of all coordinates) so the runner can
! confirm all readers reconstruct identical data. Timing inside the program
! excludes process-startup cost from the measurement.

program plot3d_read_fortran
   use, intrinsic :: iso_fortran_env, only: real32, real64, int32, int64
   implicit none

   character(len=4096) :: gridfile, arg
   integer :: nreps, rep, nargs
   integer(int64) :: c0, c1, crate
   real(real64) :: checksum, guard
   real(real64) :: secs

   nargs = command_argument_count()
   if (nargs < 2) then
      write(*,*) 'usage: plot3d_read_fortran <gridfile> <nreps>'
      stop 1
   end if
   call get_command_argument(1, gridfile)
   call get_command_argument(2, arg)
   read(arg, *) nreps

   call system_clock(count_rate=crate)

   ! Timed repetitions: read + reconstruct only.
   do rep = 1, nreps
      call system_clock(c0)
      call read_reconstruct(trim(gridfile), guard)
      call system_clock(c1)
      secs = real(c1 - c0, real64) / real(crate, real64)
      write(*, '(A,I0,1X,ES23.15)') 'rep ', rep, secs
   end do

   ! One untimed pass to produce the verification checksum.
   call read_checksum(trim(gridfile), checksum)
   write(*, '(A,ES23.15)') 'checksum ', checksum
   ! Print the guard so the compiler cannot elide the timed reads.
   write(*, '(A,ES23.15)') 'guard ', guard

contains

   subroutine open_header(fname, u, lnb, lni, lnj, lnk)
      character(len=*), intent(in) :: fname
      integer, intent(out) :: u
      integer(int32), intent(out) :: lnb
      integer(int32), allocatable, intent(out) :: lni(:), lnj(:), lnk(:)
      integer :: ier, lb
      open(newunit=u, file=fname, access='stream', form='unformatted', &
           status='old', action='read', iostat=ier)
      if (ier /= 0) then
         write(*,*) 'error opening file: ', fname
         stop 2
      end if
      read(u) lnb
      allocate(lni(lnb), lnj(lnb), lnk(lnb))
      do lb = 1, lnb
         read(u) lni(lb), lnj(lb), lnk(lb)
      end do
   end subroutine open_header

   subroutine read_reconstruct(fname, guard_out)
      ! Read the file and reconstruct each block into an (ni,nj,nk,3) array.
      ! Accumulate one element per block into guard_out so the reads cannot be
      ! optimized away. No full-array reduction here, to keep this comparable
      ! to the Python read+reshape task.
      character(len=*), intent(in) :: fname
      real(real64), intent(out) :: guard_out
      integer :: u
      integer(int32) :: lnb, lb
      integer(int32), allocatable :: lni(:), lnj(:), lnk(:)
      real(real32), allocatable :: dat(:,:,:,:)

      call open_header(fname, u, lnb, lni, lnj, lnk)
      guard_out = 0.0_real64
      do lb = 1, lnb
         allocate(dat(lni(lb), lnj(lb), lnk(lb), 3))
         ! Stream read fills the array in Fortran order: i fastest, then j, k, c.
         ! This matches the order='F' reshape used by GridIO.read_grid.
         read(u) dat
         guard_out = guard_out + real(dat(1, 1, 1, 1), real64)
         deallocate(dat)
      end do
      close(u)
      deallocate(lni, lnj, lnk)
   end subroutine read_reconstruct

   subroutine read_checksum(fname, csum)
      ! Untimed verification pass: full float64 sum over all coordinates.
      character(len=*), intent(in) :: fname
      real(real64), intent(out) :: csum
      integer :: u
      integer(int32) :: lnb, lb, i, j, k, c
      integer(int32), allocatable :: lni(:), lnj(:), lnk(:)
      real(real32), allocatable :: dat(:,:,:,:)

      call open_header(fname, u, lnb, lni, lnj, lnk)
      csum = 0.0_real64
      do lb = 1, lnb
         allocate(dat(lni(lb), lnj(lb), lnk(lb), 3))
         read(u) dat
         do c = 1, 3
            do k = 1, lnk(lb)
               do j = 1, lnj(lb)
                  do i = 1, lni(lb)
                     csum = csum + real(dat(i, j, k, c), real64)
                  end do
               end do
            end do
         end do
         deallocate(dat)
      end do
      close(u)
      deallocate(lni, lnj, lnk)
   end subroutine read_checksum

end program plot3d_read_fortran
