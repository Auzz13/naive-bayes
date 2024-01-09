# data
import pandas as pd
import numpy as np

# buat ML
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB

# grafik
import matplotlib.pyplot as plt

# %% [markdown]
# ## BACA DATASET

# %%
# baca data excel

data = pd.read_excel('import_data_awal.xlsx')  
data.head()

# %%
# mengecek anomali pada isi data (unique data setiap kolom) = tidak ada penyimpangan

print('NILAI SETIAP pclass')

for pclass in data:
    print('\n---------------\n'+pclass+'\n---------------')
    value = data[pclass].unique()
    index = 1
    for child_val in value:
        print(str(index)+'. '+str(child_val))
        index+=1

# %% [markdown]
# ## KONVERSI NILAI STRING KE DISKRIT (ANGKA)

# %%
# digunakan untuk mengkonversi nilai string ke bentuk diskrit untuk klasifikasi -> misal ya, tidak -> 1,0

# params : data, mode
# data -> ngambil data awal
# mode (default : full) -> apakah diubah semua atau tidak
# mode full -> ubah ke angka untuk training
# mode selain full -> ubah ke angka untuk deteksi

def to_diskrit(data,mode='full'):
    # mengubah usia -> 20-40 = 0, 40-50 = 1, 50-60 = 2
    data.loc[data['usia'] == '20-40', 'usia'] = 0
    data.loc[data['usia'] == '40-50', 'usia'] = 1
    data.loc[data['usia'] == '50-60', 'usia'] = 2

    # mengubah jenis kelamin -> pria = 1, wanita = 0
    data.loc[data['jkel'] == 'pria', 'jkel'] = 1
    data.loc[data['jkel'] == 'wanita', 'jkel'] = 0

    # mengubah banyak kencing -> ya = 1, tidak = 0
    data.loc[data['banyak_kencing'] == 'ya', 'banyak_kencing'] = 1
    data.loc[data['banyak_kencing'] == 'tidak', 'banyak_kencing'] = 0

    # mengubah turun bb -> ya = 1, tidak = 0
    data.loc[data['turun_bb'] == 'ya', 'turun_bb'] = 1
    data.loc[data['turun_bb'] == 'tidak', 'turun_bb'] = 0

    # mengubah luka sukar -> ya = 1, tidak = 0
    data.loc[data['luka_sukar'] == 'ya', 'luka_sukar'] = 1
    data.loc[data['luka_sukar'] == 'tidak', 'luka_sukar'] = 0

    # mengubah kesemutan -> ya = 1, tidak = 0
    data.loc[data['kesemutan'] == 'ya', 'kesemutan'] = 1
    data.loc[data['kesemutan'] == 'tidak', 'kesemutan'] = 0

    # mengubah lemas -> ya = 1, tidak = 0
    data.loc[data['lemas'] == 'ya', 'lemas'] = 1
    data.loc[data['lemas'] == 'tidak', 'lemas'] = 0

    # mengubah kulit_gatal -> ya = 1, tidak = 0
    data.loc[data['kulit_gatal'] == 'ya', 'kulit_gatal'] = 1
    data.loc[data['kulit_gatal'] == 'tidak', 'kulit_gatal'] = 0

    # mengubah keturunan -> ya = 1, tidak = 0
    data.loc[data['keturunan'] == 'ya', 'keturunan'] = 1
    data.loc[data['keturunan'] == 'tidak', 'keturunan'] = 0
    
    if(mode=='full'):
    
        # mengubah hasil -> ya = 1, tidak = 0
        data.loc[data['hasil'] == 'ya', 'hasil'] = 1
        data.loc[data['hasil'] == 'tidak', 'hasil'] = 0
        
        return data
    
    return data


# %%
# overview hasil konversi data

data = to_diskrit(data)
data.head()

# %% [markdown]
# ## MENGECEK MISSING VALUE / OUTLIERS

# %%
# mengecek nilai null pada dataset
cek_anomali = data.isnull().sum()
cek_anomali

# %% [markdown]
# ## MEMISAHKAN X dan y untuk klasifikasi

# %% [markdown]
# ### MEMISAHKAN KELAS "y"

# %%
# menghapus kolom hasil dan menampilkannya = kelas y
y = data.pop('hasil')
y.head()

# %% [markdown]
# ### MEMISAHKAN KELAS "X"

# %%
# kelas X = df data - kolom hasil
X = data
X.head()

# %% [markdown]
# # UJI MODEL NAIVE BAYES

# %% [markdown]
# ## Pembuatan model klasifikasi

# %%
# import MultinomialNB dari sklearn module
# alpha = 1 -> laplacian smoothing

model = MultinomialNB(alpha=1.0)

test_size = 0.2
# test size = sampel data dari populasi yang akan digunakan untuk melakukan prediksi terhadap data baru
# 0.2 = 20% dari populasi, misal 1000 data -> 200 data akan digunakan sebagai data training 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =test_size, random_state = 44)
# membagi data training set kedalam dataset training dan testing
# dataset training -> set data yang akan digunakan untuk pembelajaran oleh komputer
# dataset testing -> set / kumpulan data yang akan digunakan sebagai acuan prediksi data baru

# X_train -> berisikan data gejala (usia, jkel, keturunan, dll) untuk proses training
# X_test -> berisikan data gejala untuk memprediksi

# y_train -> kumpulan hasil aktual untuk proses training
# y_test -> kumpulan hasil aktual untuk memprediksi



# %%
X_train.head()
# nomor pada row acak = mengambil random set data untuk dibagi antara x_train dan x_test

# %%
X_test.head()

# %%
y_train.head()

# %%
y_test.head()

# %%
# melakukan training pada data

model.fit(X_train,y_train)

# %%
# y_pred = dataset hasil klasifikasi dari dataset testing -> digunakan untuk melihat kinerja naive bayes

y_pred = model.predict(X_test)
y_pred

# %% [markdown]
# ## Kinerja Klasifikasi

# %%
cm = confusion_matrix(y_test, y_pred)
cm

# %%
# melihat akurasi dari penggunaan data training dan testing untuk memprediksi data baru

accuracy_score(y_test, y_pred)

# %%
# mencetak hasil klasifikasi

print(classification_report(y_test, y_pred))

# %%
# test kinerja klasifikasi dengan naive bayes berdasarkan test size

def model_performance(test_size, params='all'):
    
    # setting awal training & testing set sesuai test size
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =test_size, random_state = 44)
    
    # inisiasi model dengan data training
    model.fit(X_train,y_train)
    
    # mengecek performa dengan y_pred -> confusion matrix, classification report dari prediksi X_test
    y_pred = model.predict(X_test)
    
    
    heading = str(test_size*100)+'%'
    print('\n')
    print('<--------------------- Sample test size '+heading+' --------------------->')
    
    if params == 'all' or params == 'accuracy':
        print('\n')
        print('Akurasi')
        print('======================')
        accuracy = accuracy_score(y_test, y_pred)
        print(str(round(accuracy*100,2))+'%')
            
    if params == 'all' or params == 'confusion_matrix':
        print('\n')
        print('Confusion Matrix')
        print('======================')
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
    if params == 'all' or params == 'classification_report':
        print('\n')
        print('Classification Report')
        print('======================')
        print(classification_report(y_test, y_pred))
        

# %%
# analisa hasil penggunaan test size dengan mengambil nilai akurasi & performa klasifikasi dari ukuran test size

# cetak hasil performa klasifikasi dengan sampel datasize 10% - 90%
def print_performance(params = 'all'):
    for i in range(1,10):
            value = round(i*0.1 ,1)
            # value = 0.1, 0.2, 0.3, ...., 0.9
            model_performance(value,params)
        
print_performance('all')

# %%
data_test_size = []
data_akurasi = []
for i in range(1,10):
    test_size = round(i*0.1 ,1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =test_size, random_state = 44)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    data_akurasi.append(round(accuracy*100,2))
    data_test_size.append(str(round(test_size*100,2))+'%')
    

plot = plt.bar(data_test_size, data_akurasi)

for value in plot:
    height = value.get_height()
    plt.text(value.get_x() + value.get_width()/2.,
             1.002*height,'%d' % int(height), ha='center', va='bottom')
    
plt.title("Akurasi berdasarkan test size")
plt.xlabel("Size Data Testing")
plt.ylabel("Akurasi")
plt.xticks(rotation=70)
# Display the graph on the screen
plt.show()

# %% [markdown]
# ## PEMBUATAN MODEL KLASIFIKASI

# %% [markdown]
# ## Penentuan model yang digunakan & training data

# %%
# alpha 1 -> laplacian smoothing
model = MultinomialNB(alpha=1.0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.4, random_state = 44)

# training data
model.fit(X_train,y_train)

# %% [markdown]
# ## Membaca data yang akan digunakan untuk prediksi

# %%
# baca data gejala dari excel
data_deteksi = pd.read_excel('deteksi_gejala.xlsx',header=0)

data_deteksi

# %%
# data awal deteksi -> data gejala yang digunakan untuk kebutuhan print output

data_awal_deteksi = data_deteksi.copy()

# %%
# data deteksi -> data deteksi yang digunakan untuk klasifikasi
# data dikonversi menjadi bentuk diskrit

to_diskrit(data_deteksi,'partial')
data_deteksi.head()

# %% [markdown]
# # PROSES KLASIFIKASI

# %% [markdown]
# ## Konfigurasi Output

# %%
# membuat function untuk mengembalikan dataframe testing baris ke-index -> kebutuhan klasifikasi dari set gejala

# tabulate -> library untuk print bentuk tabel seperti oracle
from tabulate import tabulate

# mengembalikan data gejala index ke-x yang ingin diprediksi
def get_testing_value(index):
    return data_deteksi.loc[[index],:]

# print data gejala index ke-x yang ingin diprediksi (dalam bentuk tabel gejala)
def print_gejala(index):
    df = data_awal_deteksi.loc[[index],:]
    pdtabulate=lambda df:tabulate(df,headers='keys',tablefmt='psql')
    print(pdtabulate(df))

# %% [markdown]
# ## Test Prediksi pada data gejala

# %%
# uji coba deteksi dari data deteksi index ke-2

new_predicted = model.predict(get_testing_value(2))
new_predicted

# %% [markdown]
# ## Prediksi dinamis sesuai data ke-x

# %%
# melakukan prediksi pada data gejala index ke-x lalu mencetak gejala dan hasilnya

def predict_diabetes(index):
    X_test = get_testing_value(index)
    result = model.predict(X_test)
    print_gejala(index)
    hasil = 'TIDAK TERDETEKSI DIABETES'
    
    # jika hasil = 1 -> terdeteksi
    if(result[0] == 1):
        hasil = '\x1b[31mTERDETEKSI DIABETES\x1b[0m'
        
    print('HASIL : '+hasil)
    print()
    print('\n')

# %% [markdown]
# ## Prediksi pada semua data gejala

# %%
# looping untuk mencetak semua data gejala yang diinputkan (sumber : dataframe yang diimport dari file excel gejala)

for i in range(len(data_deteksi)):
    predict_diabetes(i)


