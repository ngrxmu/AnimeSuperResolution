# Anime-Super-Resolution
Static comparison result 
![](./doc/res_static.jpg)
Dynamic comparison result
![](./doc/res_dynamic.gif)
## 1. Environment
python 3.8.12
pytorch 1.7.1
### install dependencies
```bash
pip install -r requirements.txt
```
## 2. Train
### 2.1 download data
Please email to me (nangongrui1999@gmail.com) if necessary.
### 2.2 convert jpg to h5
```bash
python data2h5.py
```
You can find clipped images in "./clipdata/".
You can find a h5 file in "./".
### 2.3 train
```
python train.py
```
You can find trained model in "./Models/".
You can find training log in "./".
## 3. Test
### 3.1 test the result
```bash
python test.py
```
You can find the result in "./output/".
### 3.2 show the result
Static comparison result
```bash
python show.py --mode 0 --lr ./input/Anime_425.jpg --sr ./output/Anime_425.jpg
```
Dynamic comparison result
```bash
python show.py -mode 1 --lr ./input/Anime_425.jpg --sr ./output/Anime_425.jpg
```
## 4. Repair Anime Video
### 4.1 clip video to images
```bash
python clipvideo.py --input ./xyj.mp4 --output ./xyj
```
### 4.2 test
```bash
python test.py --input ./xyj --output ./output_xyj
```
### 4.3 make images to video
```bash
python makevideo.py --input ./output_xyj --output ./output_xyj.avi
```
