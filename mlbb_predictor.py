import streamlit as st
import numpy as np
import time
import matplotlib.pyplot as plt

# ========== 1. DEFINISI MODEL ANN BACKPROPAGATION ==========
class MobileLegendsANN:
    def __init__(self, input_size, hidden_size, output_size):
        # Inisialisasi bobot dan bias secara acak
        self.w1 = np.random.uniform(size=(input_size, hidden_size))
        self.b1 = np.random.uniform(size=(1, hidden_size))
        self.w2 = np.random.uniform(size=(hidden_size, output_size))
        self.b2 = np.random.uniform(size=(1, output_size))
        
        # Dataset latihan: [KDA, WinRate(%), JumlahMatch]
        # Normalisasi: KDA (0-10), WinRate (0-100), JumlahMatch (0-1000)
        # Output: 1 = Pro, 0 = Biasa
        self.X = np.array([
            [0.8, 0.85, 0.9],   # KDA 8, WinRate 85%, Match 900 -> Pro
            [0.3, 0.40, 0.2],   # KDA 3, WinRate 40%, Match 200 -> Biasa
            [0.5, 0.60, 0.5],   # KDA 5, WinRate 60%, Match 500 -> Pro
            [0.2, 0.30, 0.1],   # KDA 2, WinRate 30%, Match 100 -> Biasa
            [0.9, 0.95, 0.95]   # KDA 9, WinRate 95%, Match 950 -> Pro
        ])
        self.y = np.array([[1], [0], [1], [0], [1]])  # Label
        
        # Untuk menyimpan history error
        self.loss_history = []
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def train(self, epochs=15000, lr=0.1):
        for epoch in range(epochs):
            # -- Forward Propagation --
            hidden_activation = np.dot(self.X, self.w1) + self.b1
            hidden_output = self.sigmoid(hidden_activation)
            
            output_activation = np.dot(hidden_output, self.w2) + self.b2
            predicted_output = self.sigmoid(output_activation)
            
            # Hitung MSE untuk logging
            mse = np.mean(np.square(self.y - predicted_output))
            self.loss_history.append(mse)
            
            # -- Backpropagation --
            error = self.y - predicted_output
            d_predicted_output = error * self.sigmoid_derivative(predicted_output)
            
            error_hidden = d_predicted_output.dot(self.w2.T)
            d_hidden = error_hidden * self.sigmoid_derivative(hidden_output)
            
            # -- Update bobot dan bias --
            self.w2 += hidden_output.T.dot(d_predicted_output) * lr
            self.b2 += np.sum(d_predicted_output, axis=0, keepdims=True) * lr
            self.w1 += self.X.T.dot(d_hidden) * lr
            self.b1 += np.sum(d_hidden, axis=0, keepdims=True) * lr
    
    def predict(self, input_data):
        # Forward propagation untuk satu data baru
        hidden_activation = np.dot(input_data, self.w1) + self.b1
        hidden_output = self.sigmoid(hidden_activation)
        output_activation = np.dot(hidden_output, self.w2) + self.b2
        predicted_output = self.sigmoid(output_activation)
        return predicted_output

# ========== 2. MEMBANGUN APLIKASI STREAMLIT ==========
st.set_page_config(page_title="MLBB Pro Predictor", page_icon="🎮")
st.title("🎮 MLBB Pro Predictor")
st.markdown("""
Aplikasi ini menggunakan **Jaringan Saraf Tiruan (Multi-Layer Perceptron)** dengan algoritma **Backpropagation**  
untuk memprediksi apakah seorang pemain *Mobile Legends: Bang Bang* termasuk kategori **Pro** atau **Biasa**.
""")

# Cache model agar tidak dilatih ulang setiap kali interaksi
@st.cache_resource
def get_trained_model():
    model = MobileLegendsANN(input_size=3, hidden_size=4, output_size=1)
    with st.spinner("Melatih model ANN dengan data pemain MLBB..."):
        model.train(epochs=20000, lr=0.1)
    st.success("Model siap digunakan!")
    return model

model = get_trained_model()

# Tampilkan grafik penurunan error selama training (opsional)
if st.checkbox("Tampilkan grafik training error (MSE)"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(model.loss_history, color='green', linewidth=1.5)
    ax.set_title("Penurunan Mean Squared Error (MSE) selama training")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)

# ========== 3. INPUT PENGGUNA DI SIDEBAR ==========
st.sidebar.header("📊 Input Data Pemain")
st.sidebar.markdown("Masukkan performa terakhir Anda (rata-rata 10 match terakhir)")

kda = st.sidebar.slider("KDA Ratio (Kill+Assist/Death)", 0.0, 10.0, 4.5, step=0.1)
winrate = st.sidebar.slider("Win Rate (%)", 0.0, 100.0, 55.0, step=1.0)
matches = st.sidebar.slider("Total Match (Ranked)", 0, 2000, 300, step=50)

# Normalisasi ke rentang 0-1 (sesuai training)
norm_kda = kda / 10.0
norm_winrate = winrate / 100.0
norm_matches = matches / 2000.0   # asumsi max 2000 match

predict_btn = st.sidebar.button("🚀 Prediksi Kategori Pemain")

# ========== 4. HASIL PREDIKSI ==========
st.subheader("🎯 Hasil Analisis")

if predict_btn:
    with st.spinner("Menganalisis dengan jaringan saraf..."):
        time.sleep(1)
        input_data = np.array([[norm_kda, norm_winrate, norm_matches]])
        prediction = model.predict(input_data)
        prob_pro = prediction[0][0] * 100
        
    # Tentukan kategori berdasarkan ambang batas (threshold = 0.5)
    if prob_pro >= 50:
        st.balloons()
        st.success("### 🏆 STATUS: PEMAIN PRO")
        st.write("Berdasarkan input Anda, model ANN memprediksi bahwa Anda termasuk pemain **PRO** Mobile Legends.")
        st.info(f"📈 Probabilitas Pro: **{prob_pro:.1f}%**")
    else:
        st.error("### 📉 STATUS: PEMAIN BIASA")
        st.write("Berdasarkan input Anda, model ANN memprediksi bahwa Anda masih dalam kategori **BIASA**. Teruslah berlatih!")
        st.info(f"📉 Probabilitas Pro: **{prob_pro:.1f}%**")
    
    # Tampilkan ringkasan input
    with st.expander("Lihat detail input Anda"):
        st.write(f"- **KDA Ratio:** {kda}")
        st.write(f"- **Win Rate:** {winrate}%")
        st.write(f"- **Total Match:** {matches}")
else:
    st.warning("Silakan masukkan data performa Anda di **sidebar kiri** lalu tekan tombol **'Prediksi Kategori Pemain'**.")
    st.image("https://cdn-icons-png.flaticon.com/512/2040/2040985.png", width=150)  # icon MLBB (opsional)
    
st.markdown("---")
st.caption("Model dilatih menggunakan dataset kecil (5 sampel). Untuk hasil lebih akurat, perbanyak data latihan dan sesuaikan fitur.")