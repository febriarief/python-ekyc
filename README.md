# python eKYC

### Overview
The main goal of this project is to check whether the given image is a real or fake face (liveness) and compare between 2 faces which is expected to have a high of accuracy and light weight. This repo is inspired by [Silent Face Anti Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) and uses some of the file and code in it, but we use our own custom model. 

### Pre-processing
- Automatically align facial landmarks before feeding them to the model for testing.
![auto align face landmark](/src/align-face.png)

### Result
- Output after model testing, returns "Real" or "Fake" with a confidence score.
![auto align face landmark](/src/result.png)


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

##### # Liveness Detection 
Place your image in `sample` folder.<br/>

Command to run:
```bash
python main.py --filename suzuka-nakamoto.jpg
```

Output:
```bash
Face detected
Image 'suzuka-nakamoto.jpg' is Real Face.
Score: 1.39.
Blur score: 345.25
Elapsed time 2.80 s
```

### Todos:
- [ ] Auto rotate image
- [x] Align face
- [x] Detect spoof image/face
- [ ] Compare between 2 faces

### Limitations
- The images provided must have a 3:4 (w:h) ratio.
- If the face is rotated, e.g 90 degrees, the depiction of facial landmarks is invalid.

### Credits
- [davisking/dlib](https://github.com/davisking/dlib)
- [minivision-ai/Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)