use dfdx::prelude::*;
use num::{Float, One, Zero};
use rand::Rng;

type Model = (
    (LinearConstConfig<17, 50>, Tanh), // 17 = 8 bit input 1 + 8 bit input 2 + 1 bit opcode (add or subtract)
    (LinearConstConfig<50, 50>, Tanh),
    (LinearConstConfig<50, 8>, Tanh), // 8 = 8 bit wrapped output
);

const BATCH_SIZE: usize = 10;

fn main() {
    let dev: Cpu = Default::default();
    let mut model = dev.build_module(Model::default());
    let mut opt = Adam::new(
        &model,
        AdamConfig {
            lr: 1e-3,
            ..Default::default()
        },
    );
    let mut grads = model.alloc_grads();
    let mut rng = rand::thread_rng();
    let mut loss_avg = ExponentialAverage::default();

    let mut iter = 0;
    // training loop
    while loss_avg.value > 1e-3 {
        iter += 1;
        // generate problem
        let (input, target) = get_data(&dev, rng.gen(), rng.gen(), rng.gen_range(0..2));
        let prediction = model.forward_mut(input.trace(grads));

        let loss = mse_loss(prediction, target.clone());
        loss_avg.update(loss.array());
        grads = loss.backward();
        if iter % BATCH_SIZE == 0 {
            println!("Loss after update {iter}: {:?}", loss_avg.value);
            opt.update(&mut model, &grads).unwrap();
            model.zero_grads(&mut grads);
        }
    }
    println!("Learned after {iter} examples",)
}

fn get_data(dev: &Cpu, n1: u8, n2: u8, op: u8) -> (Tensor1D<17>, Tensor1D<8>) {
    let (opcode, ans) = match op {
        // Addition
        0 => (1.0, n1.wrapping_add(n2)),
        // Subtraction
        1 => (-1.0, n1.wrapping_sub(n2)),
        _ => unreachable!(),
    };
    let mut p = [0.; 17];
    get_bits(n1, &mut p);
    get_bits(n2, &mut p[8..]);
    p[16] = opcode;

    let mut a = [0.; 8];
    get_bits(ans, &mut a);
    (dev.tensor(p), dev.tensor(a))
}

fn get_bits(byte: u8, slice: &mut [f32]) {
    #[allow(clippy::needless_range_loop)]
    for i in 0..8 {
        slice[i] = if byte >> i & 1 == 1 { 1.0 } else { -1.0 };
    }
}

pub struct ExponentialAverage<T: Float> {
    beta: f64,
    moment: f64,
    pub value: T,
    t: i32,
}

impl<T: Float> Default for ExponentialAverage<T> {
    fn default() -> Self {
        ExponentialAverage {
            beta: 0.999,
            moment: 0.,
            value: One::one(),
            t: 0,
        }
    }
}

impl<T: Float> ExponentialAverage<T> {
    pub fn update(&mut self, value: T) {
        self.t += 1;
        self.moment = self.beta * self.moment + (1. - self.beta) * value.to_f64().unwrap();
        // bias correction
        self.value = T::from(self.moment / (1. - f64::powi(self.beta, self.t))).unwrap();
    }

    pub fn reset(&mut self) {
        self.moment = 0.;
        self.value = Zero::zero();
        self.t = 0;
    }
}
