! plot3d_read_fortran.f90
!
! Compiled Fortran PLOT3D multi-block grid readers used as the baseline in the
! lptlib I/O benchmark. The file layout is the little-endian binary layout that
! lptlib.io.GridIO.read_grid consumes:
!
!     int32                    nb
!     int32 * (3*nb)           ni, nj, nk per block, interleaved
!     float32                  block data, each block (ni,nj,nk,3) in
!                              Fortran (column-major) order
!
! Usage:
!     plot3d_read_fortran <gridfile> <nreps> <mode> [checksum]
!
! The optional fourth argument is "nocheck" to suppress the untimed
! verification pass. The runner interleaves readers by invoking this program
! once per repetition, and asks for the checksum only on the first invocation
! so the verification pass is not paid 31 times over.
!
! MODES. Each mode names the amount of work done per repetition, so the Fortran
! baseline can be matched against the Python reader that does the same work.
! An earlier version of this benchmark only had the `raw` mode and compared it
! against Python readers that also reorder memory; that comparison was not
! like-for-like and is the reason the modes are now explicit.
!
!   raw     Stream-read each block into an (ni,nj,nk,3) single-precision array
!           in native Fortran (column-major) order. No reordering, no upcast,
!           no bounds. This is the pure I/O floor: essentially a memcpy out of
!           the page cache. It is NOT equivalent to any of the Python readers,
!           and is reported only to show how much of each reader's time is file
!           I/O rather than memory reordering.
!
!   contig  raw, plus a reorder of every block into a C-contiguous
!           single-precision array (last index varying fastest). This matches
!           the lptlib strided reader, which materializes each block with
!           numpy.ascontiguousarray, i.e. a full column-major to row-major
!           transpose of the block. Matched work, single precision.
!
!   full    raw, plus a scatter of every block into one shared padded
!           double-precision array laid out exactly like the numpy array
!           GridIO.read_grid builds -- C-order (nimax, njmax, nkmax, 3, nb),
!           declared here with the index order reversed so the Fortran memory
!           layout is byte-for-byte the same -- plus the per-block coordinate
!           min/max bounds. This matches GridIO.read_grid. Matched work, double
!           precision.
!
! Every mode accumulates a guard value that is printed, so no read or reorder
! can be optimized away. Timing uses system_clock and covers only the
! read-and-reconstruct work; process startup is outside the timed region. One
! untimed pass afterwards prints "checksum <value>", the float64 sum of all
! coordinates, which the runner uses to verify that every reader in the
! benchmark reconstructs identical data.

program plot3d_read_fortran
   use, intrinsic :: iso_fortran_env, only: real32, real64, int32, int64
   implicit none

   character(len=4096) :: gridfile, arg, mode, checkarg
   logical :: want_checksum
   integer :: nreps, rep, nargs
   integer(int64) :: c0, c1, crate
   real(real64) :: checksum, guard
   real(real64) :: secs

   nargs = command_argument_count()
   if (nargs < 2) then
      write(*,*) 'usage: plot3d_read_fortran <gridfile> <nreps> [mode] [nocheck]'
      write(*,*) 'mode is one of: raw contig full   (default raw)'
      stop 1
   end if
   call get_command_argument(1, gridfile)
   call get_command_argument(2, arg)
   read(arg, *) nreps
   mode = 'raw'
   if (nargs >= 3) call get_command_argument(3, mode)
   want_checksum = .true.
   if (nargs >= 4) then
      call get_command_argument(4, checkarg)
      if (trim(checkarg) == 'nocheck') want_checksum = .false.
   end if

   if (trim(mode) /= 'raw' .and. trim(mode) /= 'contig' .and. &
       trim(mode) /= 'full') then
      write(*,*) 'unknown mode: ', trim(mode)
      stop 1
   end if

   call system_clock(count_rate=crate)

   do rep = 1, nreps
      call system_clock(c0)
      select case (trim(mode))
      case ('raw')
         call read_raw(trim(gridfile), guard)
      case ('contig')
         call read_contig(trim(gridfile), guard)
      case ('full')
         call read_full(trim(gridfile), guard)
      end select
      call system_clock(c1)
      secs = real(c1 - c0, real64) / real(crate, real64)
      write(*, '(A,I0,1X,ES23.15)') 'rep ', rep, secs
   end do

   if (want_checksum) then
      call read_checksum(trim(gridfile), checksum)
      write(*, '(A,ES23.15)') 'checksum ', checksum
   end if
   write(*, '(A,ES23.15)') 'guard ', guard
   write(*, '(A,A)') 'mode ', trim(mode)

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

   subroutine read_raw(fname, guard_out)
      ! Pure I/O floor: stream-read each block into a native column-major
      ! single-precision array. No reordering, no upcast, no reduction.
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
         read(u) dat
         guard_out = guard_out + real(dat(1, 1, 1, 1), real64) &
                     + real(dat(lni(lb), lnj(lb), lnk(lb), 3), real64)
         deallocate(dat)
      end do
      close(u)
      deallocate(lni, lnj, lnk)
   end subroutine read_raw

   subroutine read_contig(fname, guard_out)
      ! Matched against the lptlib strided reader: read, then materialize each
      ! block as a C-contiguous single-precision array. cdat(c,k,j,i) has the
      ! same memory layout as a numpy array of shape (ni,nj,nk,3) in C order,
      ! so this performs the identical column-major to row-major transpose that
      ! numpy.ascontiguousarray performs on the order='F' reshape.
      character(len=*), intent(in) :: fname
      real(real64), intent(out) :: guard_out
      integer :: u
      integer(int32) :: lnb, lb, i, j, k, c
      integer(int32), allocatable :: lni(:), lnj(:), lnk(:)
      real(real32), allocatable :: dat(:,:,:,:)
      real(real32), allocatable :: cdat(:,:,:,:)

      call open_header(fname, u, lnb, lni, lnj, lnk)
      guard_out = 0.0_real64
      do lb = 1, lnb
         allocate(dat(lni(lb), lnj(lb), lnk(lb), 3))
         read(u) dat
         allocate(cdat(3, lnk(lb), lnj(lb), lni(lb)))
         do i = 1, lni(lb)
            do j = 1, lnj(lb)
               do k = 1, lnk(lb)
                  do c = 1, 3
                     cdat(c, k, j, i) = dat(i, j, k, c)
                  end do
               end do
            end do
         end do
         guard_out = guard_out + real(cdat(1, 1, 1, 1), real64) &
                     + real(cdat(3, lnk(lb), lnj(lb), lni(lb)), real64)
         deallocate(cdat)
         deallocate(dat)
      end do
      close(u)
      deallocate(lni, lnj, lnk)
   end subroutine read_contig

   subroutine read_full(fname, guard_out)
      ! Matched against GridIO.read_grid: read, scatter every block into one
      ! shared zero-padded double-precision array with the same memory layout
      ! numpy uses for grd of shape (nimax, njmax, nkmax, 3, nb) in C order,
      ! and reduce the per-block coordinate min/max bounds.
      character(len=*), intent(in) :: fname
      real(real64), intent(out) :: guard_out
      integer :: u
      integer(int32) :: lnb, lb, i, j, k, c
      integer(int32) :: nimax, njmax, nkmax
      integer(int32), allocatable :: lni(:), lnj(:), lnk(:)
      real(real32), allocatable :: dat(:,:,:,:)
      real(real64), allocatable :: grd(:,:,:,:,:)
      real(real64), allocatable :: gmin(:,:), gmax(:,:)
      real(real32) :: bmin(3), bmax(3), v

      call open_header(fname, u, lnb, lni, lnj, lnk)
      nimax = maxval(lni); njmax = maxval(lnj); nkmax = maxval(lnk)
      ! Index order reversed relative to numpy so that the element
      ! grd(b, c, k, j, i) sits at the same byte offset as the numpy element
      ! grd[i, j, k, c, b] of a C-ordered array.
      allocate(grd(lnb, 3, nkmax, njmax, nimax))
      grd = 0.0_real64
      allocate(gmin(3, lnb), gmax(3, lnb))

      do lb = 1, lnb
         allocate(dat(lni(lb), lnj(lb), lnk(lb), 3))
         read(u) dat
         do i = 1, lni(lb)
            do j = 1, lnj(lb)
               do k = 1, lnk(lb)
                  do c = 1, 3
                     grd(lb, c, k, j, i) = real(dat(i, j, k, c), real64)
                  end do
               end do
            end do
         end do
         ! Per-block coordinate bounds, reduced over the single-precision
         ! block exactly as read_grid does with .min(axis=(0,1,2)), then
         ! stored as double precision. A hand-rolled comparison loop over the
         ! contiguous i index is used rather than minval/maxval on the slice
         ! dat(:,:,:,c): gfortran materializes a temporary copy for that slice,
         ! which measured roughly 2.4x slower here. The baseline should use the
         ! faster of the two idioms.
         bmin = dat(1, 1, 1, :)
         bmax = dat(1, 1, 1, :)
         do c = 1, 3
            do k = 1, lnk(lb)
               do j = 1, lnj(lb)
                  do i = 1, lni(lb)
                     v = dat(i, j, k, c)
                     if (v < bmin(c)) bmin(c) = v
                     if (v > bmax(c)) bmax(c) = v
                  end do
               end do
            end do
         end do
         gmin(:, lb) = real(bmin, real64)
         gmax(:, lb) = real(bmax, real64)
         deallocate(dat)
      end do
      close(u)

      guard_out = grd(1, 1, 1, 1, 1) + sum(gmin) + sum(gmax)
      deallocate(grd, gmin, gmax)
      deallocate(lni, lnj, lnk)
   end subroutine read_full

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
