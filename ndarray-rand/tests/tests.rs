use ndarray::{Array, Array2, ArrayView1, Axis};
use ndarray_rand::rand::{distributions::Distribution, thread_rng};
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
        let a = Array::random((m, n), Uniform::new(0., 2.));
        // Higher than the length of both axes
        let n_samples = m + n + 1;

        // We don't want to deal with sampling from 0-length axes in this test
        if m != 0 {
            if !sampling_works(&a, true, Axis(0), n_samples) {
                return false;
            }
        }

        // We don't want to deal with sampling from 0-length axes in this test
        if n != 0 {
            if !sampling_works(&a, true, Axis(1), n_samples) {
                return false;
            }
        }

        true
    }
}

quickcheck! {
    fn sampling_behaves_as_expected(m: usize, n: usize, with_replacement: bool) -> bool {
        let a = Array::random((m, n), Uniform::new(0., 2.));
        let mut rng = &mut thread_rng();

        // We don't want to deal with sampling from 0-length axes in this test
        if m != 0 {
            let n_row_samples = Uniform::from(1..m+1).sample(&mut rng);
            if !sampling_works(&a, with_replacement, Axis(0), n_row_samples) {
                return false;
            }
        }

        // We don't want to deal with sampling from 0-length axes in this test
        if n != 0 {
            let n_col_samples = Uniform::from(1..n+1).sample(&mut rng);
            if !sampling_works(&a, with_replacement, Axis(1), n_col_samples) {
                return false;
            }
        }

        true
    }
}

fn sampling_works(a: &Array2<f64>, with_replacement: bool, axis: Axis, n_samples: usize) -> bool {
    let samples = a.sample_axis(axis, n_samples, with_replacement);
    samples
        .axis_iter(axis)
        .all(|lane| is_subset(&a, &lane, axis))
}

// Check if, when sliced along `axis`, there is at least one lane in `a` equal to `b`
fn is_subset(a: &Array2<f64>, b: &ArrayView1<f64>, axis: Axis) -> bool {
    a.axis_iter(axis).any(|lane| &lane == b)
}

#[test]
#[should_panic]
fn sampling_without_replacement_from_a_zero_length_axis_should_panic() {
    let n = 5;
    let a = Array::random((0, n), Uniform::new(0., 2.));
    let _samples = a.sample_axis(Axis(0), 1, false);
}

#[test]
#[should_panic]
fn sampling_with_replacement_from_a_zero_length_axis_should_panic() {
    let n = 5;
    let a = Array::random((0, n), Uniform::new(0., 2.));
    let _samples = a.sample_axis(Axis(0), 1, true);
}
