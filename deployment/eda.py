import streamlit as st
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
import random

def eda():
    st.title("Exploratory Data Analysis (EDA)")
    st.markdown("<hr>", unsafe_allow_html=True)

    base_dir = "src/data3a/training"
    classes = ["01-minor", "02-moderate", "03-severe"]
    class_names = ['Minor', 'Moderate', 'Severe']

    # Menu EDA
    eda_options = [
        "1. Contoh Gambar Tiap Kelas",
        "2. Melihat perbedaan kelas dari Visualisasi Area Gelap",
        "3. Distribusi Noise (High Pass Filter)",
        "4. Membandigkan kelas berdasarkan Noise ",
        "5. Melihat kelas kerusakan berdasakarkan Kontur dan Tekstur"
    ]
    selected_eda = st.selectbox("Pilih Analisis EDA:", eda_options)

    # 1. Contoh Gambar Tiap Kelas
    if selected_eda == eda_options[0]:
        st.subheader("Contoh Gambar Tiap Kelas")
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for i, cls in enumerate(classes):
            folder = os.path.join(base_dir, cls)
            files = [f for f in os.listdir(folder) if f.endswith(('.jpg', '.png'))]
            sample_file = random.choice(files)
            img = cv2.imread(os.path.join(folder, sample_file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[i].imshow(img)
            axes[i].set_title(class_names[i])
            axes[i].axis('off')
        st.pyplot(fig)

        st.markdown("## Kesimpulan")
        st.markdown("""
        Dari sampel gambar di atas bisa saya simpulkan  ada perbedaan yang sangat signifikan dari kerusakan mobil pada tiap kelas.
        """)

    # 2. Visualisasi Area Gelap
    elif selected_eda == eda_options[1]:
        st.subheader("Visualisasi Area Gelap")
        def dark_area_mask(img, thresh=50):
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            return (gray <= thresh).astype(float)

        for idx, cls in enumerate(classes):
            folder = os.path.join(base_dir, cls)
            files = [f for f in os.listdir(folder) if f.endswith(('.jpg', '.png'))]
            samples = random.sample(files, min(3, len(files)))
            fig, axes = plt.subplots(1, len(samples), figsize=(12, 4))
            fig.suptitle(f"Dark Area - {class_names[idx]}")
            for i, f in enumerate(samples):
                img = cv2.imread(os.path.join(folder, f))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[i].imshow(img)
                axes[i].imshow(dark_area_mask(img), cmap='Reds', alpha=0.4)
                axes[i].axis('off')
            st.pyplot(fig)

        st.markdown("## Kesimpulan")
        st.markdown("""
        1. Area gelap tidak bisa jadi ciri kerusakan dan tidak bisa membedakan secara spesifik pada tiap kelas
           Itu bisa jadi karena area gelap masuk ke dalamnya bayangan, bagian bawah mobil, atau pencahayaan (lighting) yang tidak konsisten
        2. Pemisahan kelas berdasarkan area gelap juga dipengaruhi banyak faktor eksternal seperti background dan lain-lain yang membuat kurang efektif
        """)

    # 3. Distribusi Noise
    elif selected_eda == eda_options[2]:
        st.subheader("Distribusi Noise (High Pass Filter)")
        def compute_noise(image_path):
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (224, 224))
            hp = cv2.Laplacian(img, cv2.CV_64F)
            return np.abs(hp).mean()

        noise_result = {}
        for cls in classes:
            folder = os.path.join(base_dir, cls)
            files = [f for f in os.listdir(folder) if f.endswith(('.jpg', '.png'))]
            noise_vals = [compute_noise(os.path.join(folder, f)) for f in files]
            noise_result[cls] = noise_vals

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.boxplot([noise_result[cls] for cls in classes], labels=class_names)
        ax.set_title("Distribusi Noise Tiap Kelas")
        ax.set_ylabel("Mean Intensitas Noise")
        st.pyplot(fig)

        st.markdown("## Kesimpulan")
        st.markdown("""
        1. Median noise pada gambar terlihat meningkat berbanding lurus dengan tingkat kerusakan, dimana kelas severe menampilkan noise yang cukup tinggi  
        2. Terdapat outliers noise besar pada kelas minor, bisa jadi bukan karena kerusakan yang besar tapi pencahayaan yang kurang, bisa juga karena kualitas gambar (sensor kamera)
        3. Di kelas moderate ada variasi noise yang hampir sama dengan kelas severe dan kelas minor, itu menjadikan kelas moderate ada yang kerusakannnya mendekati sever dan mendekati minor.
        4. Banyak faktor teknis sebenarnya dalam masalah noise, seperti kualitas gambar, pencahayaan dan lain-lain
        """)

    # 4. Heatmap Noise dengan Isolasi Mobil
    elif selected_eda == eda_options[3]:
        st.subheader("Noise Isolasi Mobil (Otsu + High Pass)")
        def get_foreground_mask(gray):
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
            return mask.astype(bool)

        for cls in classes:
            folder = os.path.join(base_dir, cls)
            files = [f for f in os.listdir(folder) if f.endswith(('.jpg', '.png'))]
            samples = files[:3]
            fig = plt.figure(figsize=(12, 6))
            for i, f in enumerate(samples):
                img = cv2.imread(os.path.join(folder, f))
                img = cv2.resize(img, (224, 224))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                mask = get_foreground_mask(gray)
                noise = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
                noise_masked = np.zeros_like(noise)
                noise_masked[mask] = noise[mask]
                norm_noise = (noise_masked - noise_masked.min()) / (noise_masked.max() - noise_masked.min() + 1e-10)

                plt.subplot(2, 3, i+1)
                plt.imshow(norm_noise, cmap='hot')
                plt.title(f"{cls} Heatmap")
                plt.axis('off')
            st.pyplot(fig)
            
        st.markdown("## Kesimpulan")
        st.markdown("""
        1. Pada kelas 01-minor, area noise terlihat lebih sedikit dan tersebar secara tipis pada batas kerusakan 
        2. Pada kelas 02-moderate, noise lebih banyak dan area heatmap lebih luas terutama di sekitar area benturan.
        3. Pada kelas 03-severe, heatmap noise terlihat lebih kuat lagi, dengan area yang luas dan tebal, itu mencerminkan tingkat kerusakan yang sangat parah dengan detail noise tinggi.

        Dari poin poin diatas bisa diindikasikan bahwa saya bisa menggunakan metode visualiasi noise dengan isolasi background untuk membedakan tiap kelas, 
        akan teapi memang tidak terlalu kuat terutama kalau ada faktor-faktor lain seperti pencahayaan yang kurang atau gambar yang kurang tajam yang menyebabkan noisenya besar walaupun itu kelas minor atau moderate.
        """)

    # 5. Kontur & Tekstur (LBP)
    elif selected_eda == eda_options[4]:
        st.subheader("Kontur & Tekstur Area Kerusakan (LBP)")
        radius = 3
        n_points = 8 * radius
        fig, axes = plt.subplots(len(classes), 6, figsize=(18, len(classes)*3), squeeze=False)

        for row, cls in enumerate(classes):
            folder = os.path.join(base_dir, cls)
            files = [f for f in os.listdir(folder) if f.endswith(('.jpg', '.png'))][:3]
            for col, f in enumerate(files):
                img = cv2.imread(os.path.join(folder, f))
                img = cv2.resize(img, (224, 224))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Kontur kerusakan
                _, dmg = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(dmg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                img_contour = img.copy()
                cv2.drawContours(img_contour, contours, -1, (255, 0, 0), 2)

                # LBP texture
                lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
                lbp_norm = (lbp - lbp.min()) / (lbp.max() - lbp.min())

                axes[row, col*2].imshow(cv2.cvtColor(img_contour, cv2.COLOR_BGR2RGB))
                axes[row, col*2].axis('off')
                axes[row, col*2].set_title(f"{cls} - Kontur")

                axes[row, col*2+1].imshow(lbp_norm, cmap='inferno')
                axes[row, col*2+1].axis('off')
                axes[row, col*2+1].set_title(f"{cls} - LBP")

        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("## Kesimpulan")
        st.markdown("""
        1. Perbedaan Kontur Kerusakan  
            * Minor (ringan): Garis konturnya sedikit dan area yang rusak kecil. Jadi kerusakannya memang nggak terlalu parah. Dan bisa juga terlihat di gambar satu bahwa tidak ada kontur sama sekali, malah background yang terkena garisnya.  
            * Moderate (sedang): Garis kontur mulai banyak dan menyebar di beberapa bagian mobil, ini tanda kerusakannya lebih nyata dan cukup banyak.  
            * Severe (berat): Kontur kerusakan paling banyak, bentuknya rumit, dan area rusaknya besar. Ini jelas kerusakan sangat parah.
              
        2. Pola Tekstur LBP (Hitam)
            * LBP ini menegaskan lagi pola kerusakannya
            * Minor: Pola teksturnya tidak banyak, karena area rusaknya memang terbatas.
            * Moderate dan Severe: Pola tekstur lebih banyak dan beragam, dengan intensitas yang beda-beda, artinya kerusakannya lebih rumit dan merata di banyak bagian.  


        """)

if __name__ == '__main__':
    eda()
