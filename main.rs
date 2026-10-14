use rand::RngExt;

type Vector = Vec<f64>;

#[derive(Clone, Debug)]
struct Layer {
    // weights[neuron][input]
    weights: Vec<Vector>,
    biases: Vector,
}

impl Layer {
    fn new(input_size: usize, neuron_count: usize) -> Self {
        let mut rng = rand::rng();

        Self {
            weights: (0..neuron_count)
                .map(|_| {
                    (0..input_size)
                        .map(|_| rng.random_range(-1.0..1.0))
                        .collect()
                })
                .collect(),
            biases: vec![0.0; neuron_count],
        }
    }

    fn forward(&self, input: &[f64]) -> Vector {
        self.weights
            .iter()
            .zip(&self.biases)
            .map(|(weights, bias)| {
                let sum: f64 = weights
                    .iter()
                    .zip(input.iter())
                    .map(|(w, x)| w * x)
                    .sum::<f64>()
                    + bias;

                // Sigmoid activation: output is between 0 and 1.
                1.0 / (1.0 + (-sum).exp())
            })
            .collect()
    }
}

#[derive(Clone, Debug)]
struct Network {
    layers: Vec<Layer>,
    input_size: usize,
    output_size: usize,
}

impl Network {
    fn new(input_size: usize, hidden_sizes: &[usize], output_size: usize) -> Self {
        assert!(input_size > 0 && output_size > 0);
        assert!(hidden_sizes.iter().all(|&n| n > 0));

        let mut layers = Vec::new();
        let mut previous_size = input_size;

        for &size in hidden_sizes {
            layers.push(Layer::new(previous_size, size));
            previous_size = size;
        }

        layers.push(Layer::new(previous_size, output_size));

        Self {
            layers,
            input_size,
            output_size,
        }
    }

    fn predict(&self, input: &[f64]) -> Vector {
        assert_eq!(input.len(), self.input_size);

        let mut activation = input.to_vec();

        for layer in &self.layers {
            activation = layer.forward(&activation);
        }

        activation
    }

    // Mean squared error over all training examples.
    fn error(&self, inputs: &[Vector], expected: &[Vector]) -> f64 {
        let mut total = 0.0;
        let mut count = 0usize;

        for (input, target) in inputs.iter().zip(expected.iter()) {
            let prediction = self.predict(input);

            for (predicted, wanted) in prediction.iter().zip(target.iter()) {
                total += (predicted - wanted).powi(2);
                count += 1;
            }
        }

        total / count.max(1) as f64
    }

    // Mutate each weight/bias independently with the given probability.
    fn mutate(&mut self, mutation_percent: f64, strength: f64) {
        let mut rng = rand::rng();
        let probability = (mutation_percent / 100.0).clamp(0.0, 1.0);

        for layer in &mut self.layers {
            for neuron in &mut layer.weights {
                for weight in neuron {
                    if rng.random_bool(probability) {
                        *weight += rng.random_range(-strength..=strength);
                    }
                }
            }

            for bias in &mut layer.biases {
                if rng.random_bool(probability) {
                    *bias += rng.random_range(-strength..=strength);
                }
            }
        }
    }

    // Insert an entire hidden layer immediately before the output layer.
    // Its width matches the previous input width, preserving dimensions.
    fn add_hidden_layer(&mut self) {
        if self.layers.is_empty() {
            return;
        }

        let output_layer_index = self.layers.len() - 1;
        let width = self.layers[output_layer_index].weights[0].len();

        self.layers.insert(
            output_layer_index,
            Layer::new(width, width),
        );
    }

    fn hidden_layer_count(&self) -> usize {
        self.layers.len() - 1
    }
}

struct EvolutionConfig {
    population_size: usize,
    generations: usize,
    elite_count: usize,
    mutation_percent: f64,
    mutation_strength: f64,
    new_layer_chance_percent: f64,
    target_error: f64,
}

fn train(
    inputs: &[Vector],
    expected: &[Vector],
    hidden_sizes: &[usize],
    config: &EvolutionConfig,
) -> Network {
    assert!(!inputs.is_empty());
    assert_eq!(inputs.len(), expected.len());
    assert!(config.population_size >= 2);
    assert!(config.elite_count >= 1);
    assert!(config.elite_count < config.population_size);

    let input_size = inputs[0].len();
    let output_size = expected[0].len();

    assert!(input_size > 0 && output_size > 0);

    for (input, target) in inputs.iter().zip(expected.iter()) {
        assert_eq!(input.len(), input_size);
        assert_eq!(target.len(), output_size);
        assert!(input.iter().chain(target.iter())
            .all(|x| x.is_finite() && (0.0..=1.0).contains(x)));
    }

    let mut rng = rand::rng();

    let mut population: Vec<Network> = (0..config.population_size)
        .map(|_| Network::new(input_size, hidden_sizes, output_size))
        .collect();

    let mut best_network = population[0].clone();
    let mut best_error = f64::INFINITY;

    for generation in 0..config.generations {
        // Sort best (lowest error) to worst.
        population.sort_by(|a, b| {
            a.error(inputs, expected)
                .total_cmp(&b.error(inputs, expected))
        });

        let current_error = population[0].error(inputs, expected);

        if current_error < best_error {
            best_error = current_error;
            best_network = population[0].clone();
        }

        if generation % 100 == 0 || best_error <= config.target_error {
            println!(
                "Generation {:5} | error {:.8} | hidden layers {}",
                generation,
                best_error,
                best_network.hidden_layer_count()
            );
        }

        if best_error <= config.target_error {
            break;
        }

        // Elitism: preserve the best networks without changing them.
        let mut next_population: Vec<Network> =
            population.iter()
                .take(config.elite_count)
                .cloned()
                .collect();

        while next_population.len() < config.population_size {
            // Select a parent from a random group of candidates.
            let tournament_size = 3.min(population.len());
            let mut parent_index = rng.random_range(0..population.len());

            for _ in 1..tournament_size {
                let candidate = rng.random_range(0..population.len());

                if population[candidate].error(inputs, expected)
                    < population[parent_index].error(inputs, expected)
                {
                    parent_index = candidate;
                }
            }

            let mut child = population[parent_index].clone();

            child.mutate(
                config.mutation_percent,
                config.mutation_strength,
            );

            // Structural mutation: occasionally create a new layer.
            if rng.random_bool(
                (config.new_layer_chance_percent / 100.0)
                    .clamp(0.0, 1.0)
            ) {
                child.add_hidden_layer();
            }

            next_population.push(child);
        }

        population = next_population;
    }

    // Include the final generation when choosing the returned network.
    population.sort_by(|a, b| {
        a.error(inputs, expected)
            .total_cmp(&b.error(inputs, expected))
    });

    if population[0].error(inputs, expected) < best_error {
        best_network = population[0].clone();
    }

    println!(
        "Final error: {:.8}",
        best_network.error(inputs, expected)
    );

    best_network
}

fn main() {
    // Each input array corresponds to one expected output array.
    // Replace these examples with your own training data.
    let inputs = vec![
    vec![0.0, 0.0],
    vec![0.0, 1.0],
    vec![1.0, 0.0],
    vec![1.0, 1.0],
    vec![0.2, 0.8],
    vec![0.4, 0.6],
];

let expected = vec![
    vec![0.0],
    vec![0.5],
    vec![0.5],
    vec![1.0],
    vec![0.5],
    vec![0.5],
];

    let config = EvolutionConfig {
        population_size: 200,
        generations: 20_000,
        elite_count: 10,

        // Chance of changing each individual weight or bias.
        mutation_percent: 10.0,

        // Maximum absolute change per mutation.
        mutation_strength: 0.5,

        // Chance of adding a new hidden layer to each offspring.
        new_layer_chance_percent: 0.5,

        // Stop once mean squared error reaches this value.
        target_error: 0.001,
    };

    let best = train(
        &inputs,
        &expected,
        &[8], // Initial hidden layer(s); e.g. &[8, 6].
        &config,
    );

    println!("\nPredictions:");

    for input in &inputs {
        println!(
            "{:?} -> {:?}",
            input,
            best.predict(input)
        );
    }

    println!(
        "\nHidden layers: {}",
        best.hidden_layer_count()
    );
}
