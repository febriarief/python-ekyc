# python eKYC

### Overview
The main goal of this project is to check whether the given image is a real or fake face (liveness) and compare between 2 faces which is expected to have a high of accuracy and light weight. This repo is inspired by [Silent Face Anti Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) and uses some of the file and code in it, but we use our own custom model. 

### Installation
##### Conda
```bash
conda env create -f environment.yml
```

Activate venv:
```bash
conda activate python-ekyc
```

##### Python
Tested on python 3.8
```bash
pip install -r requirements.txt
```

### Usage

Place your image in `sample` folder. Edit variable `filename` in file `main.py`
```python
# main.py

if __name__ == '__main__':
    start_time = time.time()

    filename = 'moa-kikuchi.jpg' #---> Your image's filename here
    filepath = os.path.join('sample', filename)
    img = cv2.imread(filepath)
```
run `python main.py`. Output:
```bash
(python-ekyc) D:\python-ekyc>python main.py
Face detected
Image 'moa-kikuchi.jpg' is Real Face.
Score: 1.04.
Blur score: 88.09
Elapsed time 0.54 s
```


### Todos:
- [ ] Auto rotate image
- [x] Detect spoof image/face
- [ ] Compare between 2 faces

### Credits
- [Silent Face Anti Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)