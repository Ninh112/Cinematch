
## Getting Started

Follow the steps below to clone the repository and run the movie recommendation system locally.

---

## 1. Clone the Repository

```bash
git clone git remote add origin https://github.com/Ninh112/Cinematch.git

cd Cinematch
```

---

## 2. Create and Activate Conda Environment

This project requires Python 3.11.

```bash
conda create -n cine python=3.11 -y
conda activate cine
```

---

## 3. Create Project Folders

Generate the required folder structure.

```bash
sh create.sh
```

---

## 4. Install Dependencies

Install all required packages listed in `requirements.txt`.

```bash
conda install -c conda-forge --file requirements.txt
```

---

## 5. Download the Dataset

This project uses the MovieLens 100K dataset.

```bash
cd data/raw
wget https://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip
```

---

## Setup Complete

You can now proceed to explore the data and build the movie recommendation system.


