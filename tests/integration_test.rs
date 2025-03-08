#[cfg(test)]
mod tests {
    use kerasty::{
        Activation, Dense, Device, Loss, Metric, Model, Optimizer, Result, Sequential, Tensor,
    };

    #[test]
    fn test_dense_layer() -> Result<()> {
        let mut model = Sequential::new();
        model.add(Dense::new(
            3,
            2,
            Activation::Sigmoid,
            None,
            None,
            None,
            None,
        ));
        if let Ok(res) = model.compile(
            Optimizer::SGD(0.1),
            Loss::MeanSquaredError,
            vec![Metric::MeanSquaredError],
        ) {
            println!("Model compiled successfully: {:?}", res);
        } else {
            println!("Model compilation failed");
        }
        let x = Tensor::new(&[[1f32, 2.], [3., 4.], [5., 6.]], &Device::Cpu)?;
        let y = Tensor::new(&[[1f32], [3.], [5.]], &Device::Cpu)?;
        model.fit(x.clone(), y.clone(), 1000, 1)?;
        let y_pred = model.predict(&x);
        println!("{:?}", y_pred);
        // Check x equals y
        assert_eq!(y.to_vec2::<f32>()?, y_pred.to_vec2::<f32>()?);
        Ok(())
    }
}
