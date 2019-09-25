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
        // We don't want to deal with 0-length axis in this test
        // `n` can be zero though
        if m == 0 {
            return true;
        }

        let axis = Axis(0);
        let a = Array::random((m, n), Uniform::new(0., 2.));
        let samples = a.sample_axis(axis, m + n + 1, true);
        samples.axis_iter(axis).all(|lane| is_subset(&a, &lane, axis))
    }
}

quickcheck! {
    fn sampling_behaves_as_expected(m: usize, n: usize, with_replacement: bool) -> bool {
        // We don't want to deal with 0-length axes in this test
        if m == 0 || n == 0 {
            return true;
        }

        let a = Array::random((m, n), Uniform::new(0., 2.));
        let mut rng = &mut thread_rng();

        let n_row_samples = Uniform::from(1..m+1).sample(&mut rng);
        let samples = a.sample_axis(Axis(0), n_row_samples, with_replacement);
        let sampling_rows_works = samples.axis_iter(Axis(0)).all(|lane| is_subset(&a, &lane, Axis(0)));

        let n_col_samples = Uniform::from(1..n+1).sample(&mut rng);
        let samples = a.sample_axis(Axis(1), n_col_samples, with_replacement);
        let sampling_cols_works = samples.axis_iter(Axis(1)).all(|lane| is_subset(&a, &lane, Axis(1)));

        sampling_rows_works && sampling_cols_works
    }
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
