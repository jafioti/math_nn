use dfdx::{
    losses::mse_loss,
    nn::{Linear, ModuleBuilder, ModuleMut, ReLU, Tanh},
    optim::{Momentum, Optimizer, Sgd, SgdConfig},
    shapes::{Rank2, Rank1},
    tensor::{AsArray, Cpu, SampleTensor, Tensor1D, TensorFromArray},
    tensor_ops::Backward,
};
use num::{Float, Zero};
use rand::Rng;

// first let's declare our neural network to optimze
type Mlp = (
    (Linear<18, 100>, ReLU),
    (Linear<100, 256>, ReLU),
    (Linear<256, 100>, ReLU),
    (Linear<100, 8>, Tanh),
);

fn main() {
    let dev: Cpu = Default::default();

    let mut sgd: Sgd<Mlp> = Sgd::new(SgdConfig {
        lr: 1e-3,
        momentum: Some(Momentum::Nesterov(0.9)),
        weight_decay: None,
    });

    // let's initialize our model and some dummy data
    let mut mlp: Mlp = dev.build_module();
    let mut rng = rand::thread_rng();
    let mut loss_avg = ExponentialAverage::new();

    for i in 0..1_000_000 {
        let n1 = rng.gen::<u8>();
        let n2 = rng.gen::<u8>();
        let mut op = rng.gen_range(0..1);
        if n2 == 0 && op == 3 {
            op = rng.gen_range(0..1); // Ensure no divide by 0
        }
        let (input, target) = get_data(&dev, n1, n2, op);
        let prediction = mlp.forward_mut(input.trace());
        
        let loss = mse_loss(prediction, target.clone());
        loss_avg.update(loss.array());
        println!("Loss after update {i}: {:?}", loss_avg.value);
        let gradients = loss.backward();
        sgd.update(&mut mlp, gradients).unwrap();
    }
}

fn get_data(dev: &Cpu, n1: u8, n2: u8, op: u8) -> (Tensor1D<18>, Tensor1D<8>) {
    let (opcode1, opcode2, ans) = match op {
        0 => (false, false, n1.wrapping_add(n2)),
        1 => (false, true, n1.wrapping_sub(n2)),
        2 => (true, false, n1.wrapping_mul(n2)),
        _ => (true, true, n1.wrapping_div(n2)),
    };
    let mut v = get_bits(n1).to_vec();
    v.append(&mut get_bits(n2).to_vec());
    v.insert(0, opcode2);
    v.insert(0, opcode1);
    let v: Vec<_> = v.into_iter().map(|v| if v {-1.0} else {1.0}).collect();

    let a: Vec<_> = get_bits(ans).iter().copied().map(|v| if v {-1.0} else {1.0}).collect();

    (
        dev.tensor(TryInto::<[f32; 18]>::try_into(v).unwrap()),
        dev.tensor(TryInto::<[f32; 8]>::try_into(a).unwrap()),
    )
}

fn get_bits(byte: u8) -> [bool; 8] {
    let mut arr = [false; 8];
    for i in 0..8 {
        arr[i] = byte >> i & 1 == 1;
    }

    arr
}

pub struct ExponentialAverage<T: Float> {
    beta: f64,
    moment: f64,
    pub value: T,
    t: i32
}

impl <T: Float> Default for ExponentialAverage<T> {
    fn default() -> Self {
        ExponentialAverage {
            beta: 0.99,
            moment: 0.,
            value: Zero::zero(),
            t: 0
        }
    }
}

impl <T: Float> ExponentialAverage<T> {
    pub fn new() -> Self {
        ExponentialAverage {
            beta: 0.99,
            moment: 0.,
            value: Zero::zero(),
            t: 0
        }
    }

    pub fn with_beta(beta: f64) -> Self {
        assert!((0. ..=1.).contains(&beta));
        ExponentialAverage {
            beta,
            moment: 0.,
            value: Zero::zero(),
            t: 0
        }
    }

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