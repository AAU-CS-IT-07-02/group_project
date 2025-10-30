# How to Install do-mpc

This guide explains how to set up your environment to run the **do-mpc getting started notebook**.
Here is the link to the example: https://www.do-mpc.com/en/latest/getting_started.html
---

## 1. Install Dependencies

You can install all the necessary packages using the provided `requirements.txt` file.

1. Make sure your virtual environment is activated.
2. Run the following command:

   ```bash
   pip install -r requirements.txt
   ```

---

## 2. Fixing the GIF Animation Error

**THE GIF ANIMATION ERROR IS FIXED, THIS IS JUST WHAT I DID TO FIX IT, IN CASE YOU CHECK THE FILE ONLINE**

The notebook *getting_started.ipynb* uses `matplotlib` to create a GIF animation.  
The original code tries to use `ImageMagickWriter`, which requires installing a separate program (**ImageMagick**).

A simpler, Python-only solution is to use the **Pillow** library, which is included in the `requirements.txt` file.

To fix the error, make two small changes to the **last code cell** in your notebook:

1. **Change the import line from:**

   ```python
   from matplotlib.animation import FuncAnimation, FFMpegWriter, ImageMagickWriter
   ```

   **To:**

   ```python
   from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
   ```

2. **Change the writer initialization from:**

   ```python
   gif_writer = ImageMagickWriter(fps=3)
   ```

   **To:**

   ```python
   gif_writer = PillowWriter(fps=3)
   ```

After these changes, the cell will run successfully and create the `anim.gif` file.

## 3. Results of the simulation

**In the Images folder you can find a gif with the results of the MPC example.**

![Animation of do-mpc results](Images/anim.gif)
