//! A no-std rust crate providing a heap-allocated non-resizable type-checked array.
// !
//! The following types are provided:
//! - `HeapArray` - A one dimensional heap-allocated array.
//! - `HeapArray2D` - A two dimensional heap-allocated array.
//! - `HeapArray3D` - A three dimensional heap-allocated array.
//!
//! This crate does not depend on `std` but does require `alloc` in order to create the array.
//!
//! # Examples
//! ## Creating a large array on the heap using a function.
//! ```
//! use static_array::HeapArray;
//!
//! // Creates an array 16 MB (on 64 bit systems) in size which is larger than the standard linux stack size.
//! let array: HeapArray<usize, 2000000> = HeapArray::from_fn(|i| i);
//!
//! assert_eq!(1247562, array[1247562]);
//! ```
//!
//! ## Creating a large array from the default value of a type.
//! ```
//! use static_array::HeapArray;
//!
//! let array: HeapArray<usize, 2000000> = HeapArray::default();
//!
//! assert_eq!(0, array[1247562]);
//! ```

#![no_std]

extern crate alloc;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

use core::{ptr::{NonNull, self}, alloc::Layout, mem::MaybeUninit, ops::{Index, IndexMut, Deref, DerefMut}, borrow::{Borrow, BorrowMut}};
use alloc::{boxed::Box, alloc::{alloc, handle_alloc_error}};

/// A heap allocated contiguous one dimensional array.
/// This is equivalent in layout to the type `[T; N]`.
///
/// - `T` - The type of data stored in the array.
/// - `N` - The length of the array.
///
/// # Examples
/// ## Creating a large one dimensional array on the heap using a function.
/// ```
/// use static_array::HeapArray;
///
/// // Creates an array 16 MB (on 64 bit systems) in size which is larger than the standard linux stack size.
/// let array: HeapArray<usize, 2000000> = HeapArray::from_fn(|i| i);
///
/// assert_eq!(1247562, array[1247562]);
/// ```
///
/// ## Creating a large array from the default value of a type.
/// ```
/// use static_array::HeapArray;
///
/// let array: HeapArray<usize, 2000000> = HeapArray::default();
///
/// assert_eq!(0, array[1247562]);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HeapArray<T, const N: usize> {
    /// The data stored inside the array.
    data: Box<[T; N]>,
}

impl <T, const N: usize> HeapArray<T, N> {
    /// Allocates the heap array for this type.
    unsafe fn alloc_array() -> NonNull<[T; N]> {
        let layout = Layout::new::<[T; N]>();
        
        let pointer = alloc(layout);

        if pointer.is_null() {
            handle_alloc_error(layout)
        } else {
            NonNull::new_unchecked(pointer as *mut [T; N])
        }
    }

    /// Creates a new `HeapArray` by calling a function at each index.
    ///
    /// - `f` - The function to call.
    pub fn from_fn<F: FnMut(usize) -> T>(mut f: F) -> Self {
        unsafe {
            let array = Self::alloc_array();

            let first_element = array.as_ptr() as *mut T;

            for i in 0..N {
                ptr::write(first_element.offset(i as isize), f(i));
            }

            Self {
                data: Box::from_raw(array.as_ptr())
            }
        }
    }
    
    /// Creates a new `HeapArray` by calling a function at each index in parallel.
    ///
    /// - `f` - The function to call.
    #[cfg(feature = "rayon")]
    pub fn from_fn_par<F: Fn(usize) -> T + Send + Sync>(f: F) -> Self where T: Send + Sync {
        unsafe {
            let mut array = HeapArray::alloc_array();
            
            array.as_mut().into_par_iter().enumerate().for_each(|(i, value)| {
                ptr::write_volatile(value, f(i))
            });

            Self {
                data: Box::from_raw(array.as_ptr())
            }
        }
    }

    /// Creates a new `HeapArray` from a raw pointer.
    ///
    /// - `pointer` - The pointer containing the array data.
    ///
    /// # Safety
    /// To safely call this method the following constraints must be maintained:
    /// - The pointer must not be null.
    /// - The pointer must point to a region of memory at least as large as the array size.
    /// - The portion of the region pointed to by the pointer must be initialised up to the length
    /// of the array.
    pub unsafe fn from_raw(pointer: *mut T) -> Self {
        Self {
            data: Box::from_raw(pointer as *mut [T; N])
        }
    } 

    /// Returns a raw pointer to the data contained in this array
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    /// Returns a slice containing the array elements.
    pub fn as_slice(&self) -> &[T] {
        self.data.as_slice()
    }
    
    /// Returns a raw mutable pointer to the data contained in this array
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }

    /// Returns a mutable slice containing the array elements.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.data.as_mut_slice()
    }

    /// Returns the length of the array.
    pub const fn len(&self) -> usize {
        N
    }
}

impl <T, const N: usize> HeapArray<MaybeUninit<T>, N> {
    /// Creates a new uninitialised heap array.
    pub fn new_uninit() -> Self {
        unsafe {
            let array = Self::alloc_array();

            Self {
                data: Box::from_raw(array.as_ptr())
            }
        }
    }
}

impl <T, const N: usize> Index<usize> for HeapArray<T, N> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.data.index(index)
    }
}

impl <T, const N: usize> IndexMut<usize> for HeapArray<T, N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.data.index_mut(index)
    }
}

impl <T, const N: usize> IntoIterator for HeapArray<T, N> {
    type Item = T;

    type IntoIter = <[T; N] as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        // This moves the whole array from heap to the stack and is extremely inefficient.
        // After <https://github.com/rust-lang/rust/pull/134021> is merged, this could use
        // `<Box<[T; N]> as IntoIterator>::IntoIter`, which keeps the array in place.
        <[T; N]>::into_iter(*self.data)
    }
}

impl <T, const N: usize> Deref for HeapArray<T, N> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl <T, const N: usize> DerefMut for HeapArray<T, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl <T: Default, const N: usize> Default for HeapArray<T, N> {
    fn default() -> Self {
        Self::from_fn(|_| T::default())
    }
}

impl <T, const N: usize> AsMut<[T]> for HeapArray<T, N> {
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl <T, const N: usize> AsRef<[T]> for HeapArray<T, N> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl <T, const N: usize> Borrow<[T]> for HeapArray<T, N> {
    fn borrow(&self) -> &[T] {
        self.as_slice()
    }
}

impl <T, const N: usize> BorrowMut<[T]> for HeapArray<T, N> {
    fn borrow_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl <T, const N: usize> From<[T; N]> for HeapArray<T, N> {
    fn from(value: [T; N]) -> Self {
        Self {
            data: Box::new(value)
        }
    }
}

impl <T, const N: usize> From<HeapArray<T, N>> for [T; N] {
    fn from(value: HeapArray<T, N>) -> Self {
        *value.data
    }
}

impl <T: Clone, const N: usize> TryFrom<&[T]> for HeapArray<T, N> {
    type Error = usize;

    fn try_from(value: &[T]) -> Result<Self, usize> {
        if value.len() == N {
            Ok(Self::from_fn(|i| value[i].clone()))
        } else {
            Err(N)
        }
    }
}

impl <T: Clone, const N: usize> TryFrom<&mut [T]> for HeapArray<T, N> {
    type Error = usize;

    fn try_from(value: &mut [T]) -> Result<Self, usize> {
        if value.len() == N {
            Ok(Self::from_fn(|i| value[i].clone()))
        } else {
            Err(N)
        }
    }
}

impl <T, const N: usize> From<Box<[T; N]>> for HeapArray<T, N> {
    fn from(value: Box<[T; N]>) -> Self {
        Self {
            data: value
        }
    }
}

/// A heap allocated contiguous two dimensional array.
/// This is equivalent to the type `[[T; M]; N]`.
///
/// - `T` - The type of data stored in the array.
/// - `M` - The length of the inner array.
/// - `N` - The length of the array.
///
/// # Examples
/// ## Creating a large two dimensional array on the heap using a function.
/// ```
/// use static_array::HeapArray2D;
///
/// let array: HeapArray2D<usize, 1000, 1000> = HeapArray2D::from_fn(|i, j| i * j);
///
/// assert_eq!(10000, array[100][100]);
/// ```
///
/// ## Creating a large two dimensional array from the default value of a type.
/// ```
/// use static_array::HeapArray2D;
///
/// let array: HeapArray2D<usize, 1000, 1000> = HeapArray2D::default();
///
/// assert_eq!(0, array[100][100]);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HeapArray2D<T, const M: usize, const N: usize> {
    /// The data stored inside the array.
    data: Box<[[T; M]; N]>,
}

impl <T, const M: usize, const N: usize> HeapArray2D<T, M, N> {
    /// Allocates the heap array for this type.
    unsafe fn alloc_array() -> NonNull<[[T; M]; N]> {
        let layout = Layout::new::<[[T; M]; N]>();
        
        let pointer = alloc(layout);

        if pointer.is_null() {
            handle_alloc_error(layout)
        } else {
            NonNull::new_unchecked(pointer as *mut [[T; M]; N])
        }
    }

    /// Creates a new `HeapArray2D` by calling a function at each index.
    ///
    /// - `f` - The function to call.
    pub fn from_fn<F: FnMut(usize, usize) -> T>(mut f: F) -> Self {
        unsafe {
            let array = Self::alloc_array();

            let first_element = array.as_ptr() as *mut T;

            for i in 0..N {
                for j in 0..M {
                    let index = (i * M + j) as isize;

                    ptr::write(first_element.offset(index), f(i, j));
                }
            }

            Self {
                data: Box::from_raw(array.as_ptr())
            }
        }
    }
    
    /// Creates a new `HeapArray2D` by calling a function at each index in parallel.
    ///
    /// - `f` - The function to call.
    #[cfg(feature = "rayon")]
    pub fn from_fn_par<F: Fn(usize, usize) -> T + Send + Sync>(f: F) -> Self where T: Send + Sync {
        unsafe {
            let mut array = HeapArray2D::alloc_array();
            
            array.as_mut().into_par_iter().enumerate().for_each(|(i, value)| {
                let pointer = value as *mut T;

                for j in 0..M {
                    ptr::write_volatile(pointer.offset(j as isize), f(i, j));
                }
            });

            Self {
                data: Box::from_raw(array.as_ptr())
            }
        }
    }

    /// Creates a new `HeapArray2D` from a raw pointer.
    ///
    /// - `pointer` - The pointer containing the array data.
    ///
    /// # Safety
    /// To safely call this method the following constraints must be maintained:
    /// - The pointer must not be null.
    /// - The pointer must point to a region of memory at least as large as the array size.
    /// - The portion of the region pointed to by the pointer must be initialised up to the length
    /// of the array.
    pub unsafe fn from_raw(pointer: *mut T) -> Self {
        Self {
            data: Box::from_raw(pointer as *mut [[T; M]; N])
        }
    } 

    /// Returns a raw pointer to the data contained in this array
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr() as *const T
    }

    /// Returns a slice containing the array elements.
    pub fn as_slice(&self) -> &[[T; M]] {
        self.data.as_slice()
    }
    
    /// Returns a raw mutable pointer to the data contained in this array
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr() as *mut T
    }

    /// Returns a mutable slice containing the array elements.
    pub fn as_mut_slice(&mut self) -> &mut [[T; M]] {
        self.data.as_mut_slice()
    }

    /// Returns the length of the outer array.
    pub const fn outer_len(&self) -> usize {
        N
    }
    
    /// Returns the length of the inner array.
    pub const fn inner_len(&self) -> usize {
        M
    }
}

impl <T, const M: usize, const N: usize> HeapArray2D<MaybeUninit<T>, M, N> {
    /// Creates a new uninitialised heap array.
    pub fn new_uninit() -> Self {
        unsafe {
            let array = Self::alloc_array();

            Self {
                data: Box::from_raw(array.as_ptr())
            }
        }
    }
}

impl <T, const M: usize, const N: usize> Index<usize> for HeapArray2D<T, M, N> {
    type Output = [T; M];

    fn index(&self, index: usize) -> &Self::Output {
        self.data.index(index)
    }
}

impl <T, const M: usize, const N: usize> IndexMut<usize> for HeapArray2D<T, M, N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.data.index_mut(index)
    }
}

impl <T, const M: usize, const N: usize> IntoIterator for HeapArray2D<T, M, N> {
    type Item = [T; M];

    type IntoIter = <[[T; M]; N] as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl <T, const M: usize, const N: usize> Deref for HeapArray2D<T, M, N> {
    type Target = [[T; M]];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl <T, const M: usize, const N: usize> DerefMut for HeapArray2D<T, M, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl <T: Default, const M: usize, const N: usize> Default for HeapArray2D<T, M, N> {
    fn default() -> Self {
        Self::from_fn(|_, _| T::default())
    }
}

impl <T, const M: usize, const N: usize> AsMut<[[T; M]]> for HeapArray2D<T, M, N> {
    fn as_mut(&mut self) -> &mut [[T; M]] {
        self.as_mut_slice()
    }
}

impl <T, const M: usize, const N: usize> AsRef<[[T; M]]> for HeapArray2D<T, M, N> {
    fn as_ref(&self) -> &[[T; M]] {
        self.as_slice()
    }
}

impl <T, const M: usize, const N: usize> Borrow<[[T; M]]> for HeapArray2D<T, M, N> {
    fn borrow(&self) -> &[[T; M]] {
        self.as_slice()
    }
}

impl <T, const M: usize, const N: usize> BorrowMut<[[T; M]]> for HeapArray2D<T, M, N> {
    fn borrow_mut(&mut self) -> &mut [[T; M]] {
        self.as_mut_slice()
    }
}

impl <T, const M: usize, const N: usize> From<[[T; M]; N]> for HeapArray2D<T, M, N> {
    fn from(value: [[T; M]; N]) -> Self {
        Self {
            data: Box::new(value)
        }
    }
}

impl <T, const M: usize, const N: usize> From<HeapArray2D<T, M, N>> for [[T; M]; N] {
    fn from(value: HeapArray2D<T, M, N>) -> Self {
        *value.data
    }
}

impl <T: Clone, const M: usize, const N: usize> TryFrom<&[[T; M]]> for HeapArray2D<T, M, N> {
    type Error = usize;

    fn try_from(value: &[[T; M]]) -> Result<Self, usize> {
        if value.len() == N {
            Ok(Self::from_fn(|i, j| value[i][j].clone()))
        } else {
            Err(N)
        }
    }
}

impl <T: Clone, const M: usize, const N: usize> TryFrom<&mut [[T; M]]> for HeapArray2D<T, M, N> {
    type Error = usize;

    fn try_from(value: &mut [[T; M]]) -> Result<Self, usize> {
        if value.len() == N {
            Ok(Self::from_fn(|i, j| value[i][j].clone()))
        } else {
            Err(N)
        }
    }
}

impl <T, const M: usize, const N: usize> From<Box<[[T; M]; N]>> for HeapArray2D<T, M, N> {
    fn from(value: Box<[[T; M]; N]>) -> Self {
        Self {
            data: value
        }
    }
}

/// A heap allocated contiguous three dimensional array.
/// This is equivalent to the type `[[[T; L]; M]; N]`.
///
/// - `T` - The type of data stored in the array.
/// - `L` - The length of the innermost array.
/// - `M` - The length of the inner array.
/// - `N` - The length of the array.
///
/// # Examples
/// ## Creating a large three dimensional array on the heap using a function.
/// ```
/// use static_array::HeapArray3D;
///
/// let array: HeapArray3D<usize, 200, 200, 200> = HeapArray3D::from_fn(|i, j, k| i * j + k);
///
/// assert_eq!(10100, array[100][100][100]);
/// ```
///
/// ## Creating a large three dimensional array from the default value of a type.
/// ```
/// use static_array::HeapArray3D;
///
/// let array: HeapArray3D<usize, 200, 200, 200> = HeapArray3D::default();
///
/// assert_eq!(0, array[100][100][100]);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HeapArray3D<T, const L: usize, const M: usize, const N: usize> {
    /// The data stored inside the array.
    data: Box<[[[T; L]; M]; N]>,
}

impl <T, const L: usize, const M: usize, const N: usize> HeapArray3D<T, L, M, N> {
    /// Allocates the heap array for this type.
    unsafe fn alloc_array() -> NonNull<[[[T; L]; M]; N]> {
        let layout = Layout::new::<[[[T; L]; M]; N]>();
        
        let pointer = alloc(layout);

        if pointer.is_null() {
            handle_alloc_error(layout)
        } else {
            NonNull::new_unchecked(pointer as *mut [[[T; L]; M]; N])
        }
    }

    /// Creates a new `HeapArray3D` by calling a function at each index.
    ///
    /// - `f` - The function to call.
    pub fn from_fn<F: FnMut(usize, usize, usize) -> T>(mut f: F) -> Self {
        unsafe {
            let array = Self::alloc_array();

            let first_element = array.as_ptr() as *mut T;

            for i in 0..N {
                for j in 0..M {
                    for k in 0..L {
                        let index = (i * M * L + j * L + k) as isize;

                        ptr::write(first_element.offset(index), f(i, j, k));
                    }
                }
            }

            Self {
                data: Box::from_raw(array.as_ptr())
            }
        }
    }
    
    /// Creates a new `HeapArray3D` by calling a function at each index in parallel.
    ///
    /// - `f` - The function to call.
    #[cfg(feature = "rayon")]
    pub fn from_fn_par<F: Fn(usize, usize, usize) -> T + Send + Sync>(f: F) -> Self where T: Send + Sync {
        unsafe {
            let mut array = HeapArray3D::alloc_array();
            
            array.as_mut().into_par_iter().enumerate().for_each(|(i, value)| {
                let pointer = value as *mut [T; L] as *mut T;

                for j in 0..M {
                    for k in 0..L {
                        let index = (j * L + k) as isize;

                        ptr::write_volatile(pointer.offset(index), f(i, j, k));
                    }
                }
            });

            Self {
                data: Box::from_raw(array.as_ptr())
            }
        }
    }

    /// Creates a new `HeapArray2D` from a raw pointer.
    ///
    /// - `pointer` - The pointer containing the array data.
    ///
    /// # Safety
    /// To safely call this method the following constraints must be maintained:
    /// - The pointer must not be null.
    /// - The pointer must point to a region of memory at least as large as the array size.
    /// - The portion of the region pointed to by the pointer must be initialised up to the length
    /// of the array.
    pub unsafe fn from_raw(pointer: *mut T) -> Self {
        Self {
            data: Box::from_raw(pointer as *mut [[[T; L]; M]; N])
        }
    } 

    /// Returns a raw pointer to the data contained in this array
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr() as *const T
    }

    /// Returns a slice containing the array elements.
    pub fn as_slice(&self) -> &[[[T; L]; M]] {
        self.data.as_slice()
    }
    
    /// Returns a raw mutable pointer to the data contained in this array
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr() as *mut T
    }

    /// Returns a mutable slice containing the array elements.
    pub fn as_mut_slice(&mut self) -> &mut [[[T; L]; M]] {
        self.data.as_mut_slice()
    }

    /// Returns the length of the outer array.
    pub const fn outer_len(&self) -> usize {
        N
    }
    
    /// Returns the length of the inner array.
    pub const fn inner_len(&self) -> usize {
        M
    }
    
    /// Returns the length of the innermost array.
    pub const fn innermost_len(&self) -> usize {
        L
    }
}

impl <T, const L: usize, const M: usize, const N: usize> HeapArray3D<MaybeUninit<T>, L, M, N> {
    /// Creates a new uninitialised heap array.
    pub fn new_uninit() -> Self {
        unsafe {
            let array = Self::alloc_array();

            Self {
                data: Box::from_raw(array.as_ptr())
            }
        }
    }
}

impl <T, const L: usize, const M: usize, const N: usize> Index<usize> for HeapArray3D<T, L, M, N> {
    type Output = [[T; L]; M];

    fn index(&self, index: usize) -> &Self::Output {
        self.data.index(index)
    }
}

impl <T, const L: usize, const M: usize, const N: usize> IndexMut<usize> for HeapArray3D<T, L, M, N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.data.index_mut(index)
    }
}

impl <T, const L: usize, const M: usize, const N: usize> IntoIterator for HeapArray3D<T, L, M, N> {
    type Item = [[T; L]; M];

    type IntoIter = <[[[T; L]; M]; N] as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl <T, const L: usize, const M: usize, const N: usize> Deref for HeapArray3D<T, L, M, N> {
    type Target = [[[T; L]; M]];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl <T, const L: usize, const M: usize, const N: usize> DerefMut for HeapArray3D<T, L, M, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl <T: Default, const L: usize, const M: usize, const N: usize> Default for HeapArray3D<T, L, M, N> {
    fn default() -> Self {
        Self::from_fn(|_, _, _| T::default())
    }
}

impl <T, const L: usize, const M: usize, const N: usize> AsMut<[[[T; L]; M]]> for HeapArray3D<T, L, M, N> {
    fn as_mut(&mut self) -> &mut [[[T; L]; M]] {
        self.as_mut_slice()
    }
}

impl <T, const L: usize, const M: usize, const N: usize> AsRef<[[[T; L]; M]]> for HeapArray3D<T, L, M, N> {
    fn as_ref(&self) -> &[[[T; L]; M]] {
        self.as_slice()
    }
}

impl <T, const L: usize, const M: usize, const N: usize> Borrow<[[[T; L]; M]]> for HeapArray3D<T, L, M, N> {
    fn borrow(&self) -> &[[[T; L]; M]] {
        self.as_slice()
    }
}

impl <T, const L: usize, const M: usize, const N: usize> BorrowMut<[[[T; L]; M]]> for HeapArray3D<T, L, M, N> {
    fn borrow_mut(&mut self) -> &mut [[[T; L]; M]] {
        self.as_mut_slice()
    }
}

impl <T, const L: usize, const M: usize, const N: usize> From<[[[T; L]; M]; N]> for HeapArray3D<T, L, M, N> {
    fn from(value: [[[T; L]; M]; N]) -> Self {
        Self {
            data: Box::new(value)
        }
    }
}

impl <T, const L: usize, const M: usize, const N: usize> From<HeapArray3D<T, L, M, N>> for [[[T; L]; M]; N] {
    fn from(value: HeapArray3D<T, L, M, N>) -> Self {
        *value.data
    }
}

impl <T: Clone, const L: usize, const M: usize, const N: usize> TryFrom<&[[[T; L]; M]]> for HeapArray3D<T, L, M, N> {
    type Error = usize;

    fn try_from(value: &[[[T; L]; M]]) -> Result<Self, usize> {
        if value.len() == N {
            Ok(Self::from_fn(|i, j, k| value[i][j][k].clone()))
        } else {
            Err(N)
        }
    }
}

impl <T: Clone, const L: usize, const M: usize, const N: usize> TryFrom<&mut [[[T; L]; M]]> for HeapArray3D<T, L, M, N> {
    type Error = usize;

    fn try_from(value: &mut [[[T; L]; M]]) -> Result<Self, usize> {
        if value.len() == N {
            Ok(Self::from_fn(|i, j, l| value[i][j][l].clone()))
        } else {
            Err(N)
        }
    }
}

impl <T, const L: usize, const M: usize, const N: usize> From<Box<[[[T; L]; M]; N]>> for HeapArray3D<T, L, M, N> {
    fn from(value: Box<[[[T; L]; M]; N]>) -> Self {
        Self {
            data: value
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{HeapArray, HeapArray2D, HeapArray3D};

    #[test]
    fn test_from_fn() {
        let array: HeapArray<usize, 3> = HeapArray::from_fn(|i| i * 10);

        assert_eq!(0, array[0]);
        assert_eq!(10, array[1]);
        assert_eq!(20, array[2]);
    }
    
    #[test]
    #[cfg(feature = "rayon")]
    fn test_from_fn_par() {
        let array: HeapArray<usize, 100> = HeapArray::from_fn_par(|i| i * 10);

        assert_eq!(0, array[0]);
        assert_eq!(10, array[1]);
        assert_eq!(20, array[2]);
    }

    #[test]
    fn test_from_fn_2d() {
        let array: HeapArray2D<usize, 3, 3> = HeapArray2D::from_fn(|i, j| i * 10 + j);

        assert_eq!(0, array[0][0]);
        assert_eq!(1, array[0][1]);
        assert_eq!(2, array[0][2]);
        assert_eq!(10, array[1][0]);
        assert_eq!(11, array[1][1]);
        assert_eq!(12, array[1][2]);
        assert_eq!(20, array[2][0]);
        assert_eq!(21, array[2][1]);
        assert_eq!(22, array[2][2]);
    }
    
    #[test]
    #[cfg(feature = "rayon")]
    fn test_from_fn_par_2d() {
        let array: HeapArray2D<usize, 300, 300> = HeapArray2D::from_fn_par(|i, j| i * 10 + j);

        assert_eq!(0, array[0][0]);
        assert_eq!(1, array[0][1]);
        assert_eq!(2, array[0][2]);
        assert_eq!(10, array[1][0]);
        assert_eq!(11, array[1][1]);
        assert_eq!(12, array[1][2]);
        assert_eq!(20, array[2][0]);
        assert_eq!(21, array[2][1]);
        assert_eq!(22, array[2][2]);
    }

    #[test]
    fn test_from_fn_3d() {
        let array: HeapArray3D<usize, 2, 3, 3> = HeapArray3D::from_fn(|i, j, k| i * 10 + j + k * 100);

        assert_eq!(0, array[0][0][0]);
        assert_eq!(1, array[0][1][0]);
        assert_eq!(2, array[0][2][0]);
        assert_eq!(10, array[1][0][0]);
        assert_eq!(11, array[1][1][0]);
        assert_eq!(12, array[1][2][0]);
        assert_eq!(20, array[2][0][0]);
        assert_eq!(21, array[2][1][0]);
        assert_eq!(22, array[2][2][0]);
        
        assert_eq!(100, array[0][0][1]);
        assert_eq!(101, array[0][1][1]);
        assert_eq!(102, array[0][2][1]);
        assert_eq!(110, array[1][0][1]);
        assert_eq!(111, array[1][1][1]);
        assert_eq!(112, array[1][2][1]);
        assert_eq!(120, array[2][0][1]);
        assert_eq!(121, array[2][1][1]);
        assert_eq!(122, array[2][2][1]);
    }
    
    #[test]
    #[cfg(feature = "rayon")]
    fn test_from_fn_par_3d() {
        let array: HeapArray3D<usize, 200, 300, 300> = HeapArray3D::from_fn_par(|i, j, k| i * 10 + j + k * 100);

        assert_eq!(0, array[0][0][0]);
        assert_eq!(1, array[0][1][0]);
        assert_eq!(2, array[0][2][0]);
        assert_eq!(10, array[1][0][0]);
        assert_eq!(11, array[1][1][0]);
        assert_eq!(12, array[1][2][0]);
        assert_eq!(20, array[2][0][0]);
        assert_eq!(21, array[2][1][0]);
        assert_eq!(22, array[2][2][0]);
        
        assert_eq!(100, array[0][0][1]);
        assert_eq!(101, array[0][1][1]);
        assert_eq!(102, array[0][2][1]);
        assert_eq!(110, array[1][0][1]);
        assert_eq!(111, array[1][1][1]);
        assert_eq!(112, array[1][2][1]);
        assert_eq!(120, array[2][0][1]);
        assert_eq!(121, array[2][1][1]);
        assert_eq!(122, array[2][2][1]);
    }

    #[test]
    fn test_write_index() {
        let mut array: HeapArray<usize, 5> = HeapArray::default();

        array[3] = 5;

        assert_eq!(0, array[2]);
        assert_eq!(5, array[3]);
    }
    
    #[test]
    fn test_write_index_2d() {
        let mut array: HeapArray2D<usize, 2, 5> = HeapArray2D::default();

        array[3] = [1, 2];

        assert_eq!(0, array[0][0]);
        assert_eq!(1, array[3][0]);
        assert_eq!(2, array[3][1]);
    }
    
    #[test]
    fn test_write_index_subarray_2d() {
        let mut array: HeapArray2D<usize, 2, 5> = HeapArray2D::default();

        array[3] = [1, 2];

        assert_eq!([0, 0], array[0]);
        assert_eq!([1, 2], array[3]);
    }
    
    #[test]
    fn test_write_index_3d() {
        let mut array: HeapArray3D<usize, 3, 2, 5> = HeapArray3D::default();

        array[3] = [[1, 2, 3], [4, 5, 6]];

        assert_eq!(0, array[0][0][0]);
        assert_eq!(1, array[3][0][0]);
        assert_eq!(6, array[3][1][2]);
    }
    
    #[test]
    fn test_write_index_subarray_3d() {
        let mut array: HeapArray3D<usize, 3, 2, 5> = HeapArray3D::default();

        array[3] = [[1, 2, 3], [4, 5, 6]];

        assert_eq!([[0, 0, 0], [0, 0, 0]], array[0]);
        assert_eq!([1, 2, 3], array[3][0]);
        assert_eq!([4, 5, 6], array[3][1]);
    }
    
    #[test]
    #[should_panic]
    fn test_read_out_of_bounds() {
        let array: HeapArray<usize, 5> = HeapArray::default();

        assert_ne!(0, array[5]);
    }
    
    #[test]
    #[should_panic]
    fn test_read_out_of_bounds_2d() {
        let array: HeapArray2D<usize, 6, 5> = HeapArray2D::default();

        assert_ne!(0, array[5][0]);
    }
   
    #[test]
    #[should_panic]
    fn test_read_out_of_bounds_subarray_2d() {
        let array: HeapArray2D<usize, 6, 5> = HeapArray2D::default();

        assert_ne!([0, 0, 0, 0, 0, 0], array[5]);
    }
    
    #[test]
    #[should_panic]
    fn test_read_out_of_bounds_3d() {
        let array: HeapArray3D<usize, 7, 6, 5> = HeapArray3D::default();

        assert_ne!(0, array[5][0][0]);
    }
   
    #[test]
    #[should_panic]
    fn test_read_out_of_bounds_subarray_3d() {
        let array: HeapArray3D<usize, 2, 6, 5> = HeapArray3D::default();

        assert_ne!([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], array[5]);
    }
}

