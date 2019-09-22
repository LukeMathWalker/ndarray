use ndarray::{Array, Axis};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use quickcheck::quickcheck;

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
fn oversampling_without_replacement_should_panic() {
    let m = 5;
    let a = Array::random((m, 4), Uniform::new(0., 2.));
    let _samples = a.sample_axis(Axis(0), m + 1, false);
}

quickcheck! {
    fn oversampling_with_replacement_is_fine(m: usize, n: usize) -> bool {
        // We don't want to deal with 0-length axis in this test
        // `n` can be zero though
        if m == 0 {
            return true;
        }

        let a = Array::random((m, n), Uniform::new(0., 2.));
        let _samples = a.sample_axis(Axis(0), m + n + 1, true);
        true
    }
}

#[test]
#[should_panic]
fn sampling_from_a_zero_length_axis_should_panic() {
    let n = 5;
    let a = Array::random((0, n), Uniform::new(0., 2.));
    let _samples = a.sample_axis(Axis(0), 1, false);
}
