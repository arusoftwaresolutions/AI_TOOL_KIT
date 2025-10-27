# Mastering the AI Toolkit
Course: AI Tools and Applications  
Theme: "Mastering the AI Toolkit"  

---

## Part 1 — Theoretical Understanding

### Q1: Primary differences between TensorFlow and PyTorch. When to choose one over the other?
TensorFlow and PyTorch are leading deep learning frameworks with overlapping capabilities but different design philosophies.

- Programming model:
  - TensorFlow: Historically graph-based (static graphs), now with eager execution via TF 2.x; emphasizes production-readiness and deployment.
  - PyTorch: Dynamic computation graph by default (eager), more Pythonic and preferred for research.

- Ecosystem and tooling:
  - TensorFlow: Extensive production ecosystem (SavedModel, TF Serving, TFLite, TF.js).
  - PyTorch: Strong research community and growing production tools (TorchServe).

When to choose:
- TensorFlow for production/deployment needs and cross-platform support.
- PyTorch for rapid prototyping and research.

### Q2: Two use cases for Jupyter Notebooks in AI development
1. Exploratory data analysis and visualization.
2. Prototyping, reporting, and sharing reproducible experiments.

### Q3: How does spaCy enhance NLP tasks compared to basic Python string operations?
spaCy provides accurate tokenization, POS tagging, lemmatization, NER, and fast pipelines, which are language-aware and robust versus naive string processing.

### Comparative Analysis: Scikit-learn vs TensorFlow
- Target applications:
  - Scikit-learn: Classical ML (tabular data, clustering, feature-based methods).
  - TensorFlow: Deep learning (images, audio, sequence models).
- Ease of use:
  - Scikit-learn: Simpler API; friendly for beginners.
  - TensorFlow: More complex; Keras eases usage.
- Community support:
  - Both large; TensorFlow offers broader production tooling.

---

## Part 2 — Practical Implementation (Summary)

Files:
- task1_iris_sklearn.py — Iris Decision Tree with evaluation and confusion matrix.
- task2_mnist_cnn.py — CNN for MNIST, training and validation graphs, sample predictions, saves mnist_cnn_model.h5.
- task3_spacy_ner_sentiment.py — spaCy NER and rule-based sentiment on sample Amazon-like reviews.
- streamlit_mnist_app.py — Streamlit web UI for MNIST model upload and prediction.

Datasets used: Iris and MNIST (public). Amazon review examples are embedded.

---

## Part 3 — Ethics & Troubleshooting

### Ethical considerations and potential biases
- MNIST bias: limited handwriting styles; mitigate via augmentation and diverse data.
- Reviews bias: rule-based sentiment and NER can misinterpret slang/sarcasm; mitigate via annotated domain data and human-in-the-loop.

Tools: TensorFlow Fairness Indicators, spaCy custom filters and retraining.

### Troubleshooting Challenge — Buggy vs Fixed TensorFlow snippet

Buggy example:
```python
import tensorflow as tf
x = tf.random.normal((32, 28, 28))           # missing channel dimension
y = tf.keras.utils.to_categorical(range(32), 10)
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='mean_squared_error')  # wrong loss for classification
model.fit(x, y, epochs=1)
```

Fixed:
```python
import tensorflow as tf
import numpy as np
x = tf.random.normal((32, 28, 28, 1))
y = tf.keras.utils.to_categorical(np.random.randint(0,10,size=(32,)), 10)
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x, y, epochs=1)
```
Explanation: add channel dim, create correct one-hot labels, use categorical_crossentropy.

---

## Bonus — Deployment (Streamlit)
- File: streamlit_mnist_app.py (uses mnist_cnn_model.h5 saved by task2).
- Run: pip install streamlit tensorflow pillow
- Then: streamlit run streamlit_mnist_app.py

---

## Run instructions (step-by-step)

1. Open Windows CMD or PowerShell in c:\Users\admin\AI_tool_kit.
2. Create virtual env (optional):
   python -m venv venv
   venv\Scripts\activate
3. Install requirements:
   pip install -r requirements.txt
4. Run tasks:
   - Task 1: python task1_iris_sklearn.py
   - Task 2: python task2_mnist_cnn.py  (may take several minutes; GPU recommended)
   - Task 3: python task3_spacy_ner_sentiment.py
   - Bonus app: streamlit run streamlit_mnist_app.py

---

## References
- scikit-learn, TensorFlow/Keras, spaCy, Streamlit documentation.

