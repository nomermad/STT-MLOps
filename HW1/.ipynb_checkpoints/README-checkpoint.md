# STT-MLOps

# Model Descriptions

| Version | File Name             | Description                     |
|---------|------------------------|---------------------------------|
| v1      | model_v1.py            | Single Linear Regression using 1 x (`x4`) |
| v2      | model_v2.py            | Two X Linear Regression using (`x2`), (`x4`) |
| best    | model_v2.py            | Best-performing model based on R²    |


# Model Explanation
This model explores linear regression using the sampregdata dataset. The dataset contains 4 x columns and one y column. The first model was created using a singular X that had the strongest fit. `x4` was the strongest predictor, with a r² score of 0.2971 on the testing data. A second model was created using two X columns, which were `x2` and `x4`. These x’s were chosen, as they had the two highest R² scores of 0.2971 and 0.2179. Model two had a r² score of 0.54, which shows improved performance compared to the first model. Version control was implemented through github to demonstrate both models. Model two is referred to as the current model due to its higher accuracy. 

# Run the best model

```bash
python HW1/model_v2.py