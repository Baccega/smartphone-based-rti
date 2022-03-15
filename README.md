# 🐍📷  Smartphone Based RTI

**Reflectance Transformation Imaging** (RTI) using footage of two smartphones without requiring an expensive light dome, created in **Python** utilizing **OpenCV** .

### 🎥 Input:

![sample_input](./docs/sample_input.gif)

> Footage by professor Filippo Bergamasco (Ca Foscari University of Venice)

### 🕹 Output (interactive):

![sample_output](./docs/sample_output.gif)


This project is the assignment for the course **Geometric and 3D Computer Vision 2020/2021**.

See **[FinalProject.pdf](FinalProject.pdf)** for more details on the assignment and to download the required assets. 

## 🔧 Usage

Before running the scripts you need to download the required assets, the assets should include: 
- The calibration videos for both cameras
- The footage from the static camera
- The footage from the moving camera

You need extract them inside a new folder called `assets` in the root of the project.
Your folder structure should look like this:

![folder_structure](./docs/folder_structure.png)

> If you want you can change the location of the input files by changing the corresponding row on the file `constants.py` under the heading: `ASSETS FILE NAMES AND DELAY BETWEEN FOOTAGE`.

After that you can just run this commands and follow the TUI: 

```bash
python3 camera_calibrator.py        # Get camera intrinsics
python3 analysis.py                 # Get data from footage
python3 interactive_relighting.py   # View output
```

## ⚙️ Interpolation methods available:

- **Linear RBF** (_From the scipy library_)
- **Polinomial Texture Maps** (_Based on the homonymous paper from: Tom Malzbender, Dan Gelb, Hans Wolters_)
## 🔬 Analysis debug modes descriptions:

| #   | Mode name     | Features                                                                         |
| --- | ------------- | -------------------------------------------------------------------------------- |
| 0   | No debug      | -                                                                                |
| 1   | Minimal debug | Live footage, Current light direction, marker's contours                         |
| 2   | Full debug    | Minimal debug, Moving camera threshold, Warped moving frame, highlighted corners |

