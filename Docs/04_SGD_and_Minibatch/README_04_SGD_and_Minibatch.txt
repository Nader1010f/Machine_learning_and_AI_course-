# 📘 بخش چهارم: گرادیان نزولی تصادفی (SGD) و مینی‌بچ

در این بخش با روش‌های مدرن‌تر آموزش مدل‌ها آشنا می‌شویم:  
گرادیان نزولی تصادفی (Stochastic Gradient Descent) و مینی‌بچ (Mini-Batch Gradient Descent)

---

## ⚠️ مشکل گرادیان نزولی کامل (Batch Gradient Descent)

در روش کلاسیک، گرادیان با استفاده از تمام داده‌ها محاسبه می‌شود:

\[
w := w - \eta \cdot \nabla J(w; X_{\text{کل}})
\]

❌ ایرادات:
- محاسبات سنگین برای داده‌های بزرگ
- سرعت پایین
- نیاز به حافظه زیاد

---

## ✅ روش‌های سریع‌تر: SGD و Mini-Batch

| روش       | تعداد نمونه در هر گام | مزایا و معایب |
|-----------|------------------------|----------------|
| **SGD**   | ۱ نمونه                 | سریع ولی پرنوسان |
| **Mini-Batch** | چند نمونه (مثلاً 16 یا 64) | سریع، پایدار، مناسب GPU |

---

## 🔢 کد پایتون: مقایسه روش‌ها

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
x = np.random.rand(100)
y = 2 * x + np.random.randn(100) * 0.1

def compute_mse(w, b, x, y):
    return np.mean((w * x + b - y) ** 2)

def sgd(x, y, epochs=100, lr=0.1):
    w, b = 0.0, 0.0
    losses = []
    for epoch in range(epochs):
        for i in range(len(x)):
            xi = x[i]
            yi = y[i]
            y_pred = w * xi + b
            error = y_pred - yi
            w -= lr * error * xi
            b -= lr * error
        losses.append(compute_mse(w, b, x, y))
    return w, b, losses
```

```python
def minibatch(x, y, batch_size=16, epochs=100, lr=0.1):
    w, b = 0.0, 0.0
    losses = []
    n = len(x)
    for epoch in range(epochs):
        indices = np.random.permutation(n)
        for i in range(0, n, batch_size):
            idx = indices[i:i+batch_size]
            xb = x[idx]
            yb = y[idx]
            y_pred = w * xb + b
            error = y_pred - yb
            w -= lr * np.mean(error * xb)
            b -= lr * np.mean(error)
        losses.append(compute_mse(w, b, x, y))
    return w, b, losses
```

---

## 📈 رسم نمودار مقایسه

```python
w_sgd, b_sgd, loss_sgd = sgd(x, y, epochs=50)
w_mb, b_mb, loss_mb = minibatch(x, y, batch_size=16, epochs=50)

plt.plot(loss_sgd, label='SGD')
plt.plot(loss_mb, label='Mini-Batch')
plt.xlabel('تکرار')
plt.ylabel('MSE')
plt.title('مقایسه کاهش خطا در SGD و Mini-Batch')
plt.legend()
plt.grid(True)
plt.show()
```

---

## 🧠 نتیجه‌گیری

- **SGD** سریع ولی پرنوسان است
- **Mini-Batch** گزینهٔ استاندارد در یادگیری ماشین مدرن است (مخصوصاً با GPU)
- گرادیان نزولی کامل (BGD) فقط برای مسائل کوچک مناسب است

---