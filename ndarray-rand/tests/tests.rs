use ndarray::{Array, Axis};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

#[test]
fn test_dim() {
    let (mm, nn) = (5, 5);
    for m in 0..mm {
        for n in 0..nn {
            let a = Array::random((m, n), Uniform::new(0., 2.));
            assert_eq!(a.shape(), &[m, n]);
            assert!(a.iter().all(|x| *x < 2.));
            assert!(a.iter().all(|x| *x >= 0.));
        }
    }
}

#[test]
#[should_panic]
fn test_you_cannot_sample_without_replacement_more_rows_than_the_rows_in_the_original_array() {
    let m = 5;
    let n = 5;
    let a = Array::random((m, n), Uniform::new(0., 2.));
    let samples = a.sample_axis(Axis(0), m+1, false);
}
