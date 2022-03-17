from turtle import position
from flask import Flask, render_template, request
from markupsafe import escape
import chess
import chess.svg
import pscan
import logging
import json

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

@app.route('/')
def index():
    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    board_svg = chess.svg.board(board)
    return render_template('index.html', board=board_svg)

@app.route('/scan', methods=['GET'])
def scan():
    fen = request.args.get('fen', False)
    if fen:
        #board = chess.Board(fen)
        board = pscan.ScanBoard(fen)
        # TODO: Sanity checks for termination conditions, e.g. checkmate, stalemate, repetitions
        if not board.is_valid():
            # TODO: Handle FEN string with invalid position
            return "Loaded FEN string with invalid position"
        board_svg = chess.svg.board(board, orientation=board.turn)
        position_scanner = pscan.PositionScanner()
        features = position_scanner.list_features()
        feature_shortcuts = json.dumps(sorted([shortcut for shortcut in features]))
        square_shortcuts = json.dumps(sorted([' '.join([c.lower() for c in sq]) for sq in chess.SQUARE_NAMES]))
        results = position_scanner.analyze(board)
        json_results = json.dumps(results)
        return render_template('scan.html', board=board_svg, turn=board.turn, results=results, json_results=json_results,\
            feature_shortcuts=feature_shortcuts, square_shortcuts=square_shortcuts, features=features, json_features=json.dumps(features))
    else:
        # TODO: Handle missing FEN string
        return "Missing FEN string"

@app.route('/features', methods=['GET'])
def features():
    position_scanner = pscan.PositionScanner()
    return render_template('features.html', features=position_scanner.list_features())

@app.route('/log', methods=['GET'])
def log():
    return render_template('log.html')

if __name__ == "__main__":
    app.run(debug=True)