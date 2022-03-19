# pscan
A simple tool to train visual scanning of chess positions

## Goal
As a proxy for improving your evaluation skills, this tool aims at improving your awareness to a number of relevant features of a chess position. Indirectly, it´s been an opportunity for me to become familiar with python-chess.

## How to
* Set a position in the board and force yourself to enumerate its relevant features, trying to match those identified by the tool.
* Once you load the FEN for a position, you can see 3 columns of features to be guessed: _good for the opponent_ (or bad for you), _neutral_, and _good for you_ (or good for the opponent).
* To guess a feature, start by typing the feature's acronym (e.g. `A` `P` `N` for Absolute Pin), followed by the relevant squares. To input the squares you can either type the coordinates (e.g. `e` `5`), or directly click on the board. Finally, press `Enter` to let the tool evaluate your answer.
* Correct answers will be gradually revealed on screen. Once a feature is guessed completely (e.g. all passed pawns for a player), the corresponding visual annotation is displayed on the board when you hover over the feature.

## Disclaimer
This project is a work in progress, mainly for personal use. As of today, it is not structured as a Python package, so there is no clean way to install it. You will need to manually setup a Python environment where both [Flask](https://flask.palletsprojects.com/en/2.0.x/) and [python-chess](https://python-chess.readthedocs.io/en/stable/index.html) are installed, and then run the main file `app.py`.