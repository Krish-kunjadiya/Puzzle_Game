# Hand-Controlled Puzzle Game 🖐️🧩

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

An interactive puzzle game where you use your hand gestures, tracked in real-time via your webcam, to solve puzzles. This project leverages computer vision to create an intuitive and fun human-computer interface, completely eliminating the need for a mouse or keyboard.

---

## **Features**

-   **Real-Time Hand Tracking:** Utilizes MediaPipe to accurately detect hand landmarks from a live webcam feed.
-   **Intuitive Gesture Controls:**
    -   **Pinch to Grab:** Bring your thumb and index finger together to pick up a puzzle piece.
    -   **Drag & Drop:** Move your hand to drag the selected piece across the screen.
    -   **Release to Place:** Separate your fingers to drop the piece. It automatically snaps into the correct grid slot if placed correctly.
-   **Dynamic Level System:** Progress through 5 levels of increasing difficulty, from a 2x2 grid to a challenging 6x6 grid.
-   **Randomized Puzzles:** Each level loads a new, random image from a local folder, ensuring no two playthroughs are the same.
-   **Gesture-Based Navigation:** After winning a level, use hand gestures to navigate the menu:
    -   ✌️ **Two Fingers ("Peace"):** Advance to the next level.
    -   👍 **Thumbs Up:** Exit the game.
-   **Polished UI:** A clean and beautiful interface with semi-transparent panels for the timer and level display, and a stylish win screen.

---

## **Tech Stack**

-   **Python:** Core programming language.
-   **OpenCV:** For handling the webcam feed, image processing, and drawing the UI.
-   **MediaPipe:** For robust, high-fidelity hand and gesture tracking.
-   **NumPy:** For efficient numerical operations on image data.

---

## **Setup and Installation**

Follow these steps to get the project running on your local machine.

### **Prerequisites**
- Python 3.8+
- A webcam

### **Installation Steps**

1.  **Clone the repository:**
    ```bash
    https://github.com/Krish-kunjadiya/Puzzle_Game.git
    cd Puzzle_Game
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\Activate.ps1

    # For macOS / Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Add your puzzle images:**
    -   Create a folder named `puzzle_images` in the root of the project directory.
    -   Add several `.jpg` or `.png` images to this folder. The game will pick from these randomly.

---

## **How to Run**

Once the setup is complete, run the following command in your terminal:
```bash
python puzzle_game.py
```

The game window will open, and you can start playing!

---

## **Future Improvements**

- [ ] Add sound effects for grabbing, dropping, and winning.
- [ ] Implement a high-score system to save best times per level.
- [ ] Allow the user to select a specific puzzle image from the folder.
- [ ] Introduce different puzzle shapes or rotating pieces for an extra challenge.

---

## **License**

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## **Acknowledgments**

A big thank you to the developers of **Google's MediaPipe** for creating such a powerful and accessible tool for hand tracking.
