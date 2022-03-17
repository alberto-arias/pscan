from abc import ABC, abstractmethod
import logging
import chess
import chess.svg
from enum import IntEnum
import copy
from itertools import accumulate

all_pieces_and_pawns = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
all_pieces = [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
minor_pieces = [chess.KNIGHT, chess.BISHOP]
major_pieces = [chess.ROOK, chess.QUEEN]
long_range_pieces = [chess.BISHOP, chess.ROOK, chess.QUEEN]

def _clip_to_board(index:int) -> int:
    return 0 if index < 0 else (7 if index > 7 else index)

class AdjacentFilesIterator:
    def __init__(self, file:int) -> None:
        self._file = _clip_to_board(file - 1)
        self._end = file + 2

    def __iter__(self):
        return self

    def __next__(self) -> int:
        if self._file == self._end:
            raise StopIteration
        else:
            self._file += 1
            return self._file - 1

class AllRanksIterator:
    def __init__(self, rank:int, color: chess.Color) -> None:
        self._rank = 0
        self._end = 8
        self._color = color

    def __iter__(self):
        return self

    def __next__(self) -> int:
        if self._rank == self._end:
            raise StopIteration
        else:
            self._rank += 1
            return self._rank - 1

class InFrontRanksIterator:
    def __init__(self, rank:int, color: chess.Color) -> None:
        self._rank = _clip_to_board(rank + 1 if color else rank - 1)
        self._end = 8 if color else -1
        self._color = color

    def __iter__(self):
        return self

    def __next__(self) -> int:
        if self._rank == self._end:
            raise StopIteration
        elif self._color:
            self._rank += 1
            return self._rank - 1
        else:
            self._rank -= 1
            return self._rank + 1

class BehindAndEqualRanksIterator:
    def __init__(self, rank:int, color: chess.Color) -> None:
        self._rank = _clip_to_board(rank)
        self._end = -1 if color else 8
        self._color = color

    def __iter__(self):
        return self

    def __next__(self) -> int:
        if self._rank == self._end:
            raise StopIteration
        elif self._color:
            self._rank -= 1
            return self._rank + 1
        else:
            self._rank += 1
            return self._rank - 1

def _filter_bitboard_index(index:int) -> bool:
    return index >= 0 and index < 8

def _init_pawn_lookup_tables(file_iterator, rank_iterator):
    ret = {chess.WHITE: {}, chess.BLACK: {}}
    logging.debug("BEGIN - Generating BB_BACKWARD_PAWN look-up tables")
    for color in [chess.WHITE, chess.BLACK]:
        for rank in range(0, 8):
            for file in range(0, 8):
                file_mask = accumulate([chess.BB_FILES[f] for f in filter(_filter_bitboard_index, file_iterator(file))], lambda a, b: a | b)
                rank_mask = accumulate([chess.BB_RANKS[r] for r in filter(_filter_bitboard_index, rank_iterator(rank, color))], lambda a, b: a | b)
                for fm in file_mask:
                    pass
                for rm in rank_mask:
                    pass
                ret[color][int(chess.SquareSet([chess.square(file_index=file, rank_index=rank)]))] = fm & rm
    logging.debug("END - Generating BB_BACKWARD_PAWN look-up tables")
    return ret

BB_BACKWARD_PAWN = _init_pawn_lookup_tables(AdjacentFilesIterator, BehindAndEqualRanksIterator)
BB_PASSED_PAWN = _init_pawn_lookup_tables(AdjacentFilesIterator, InFrontRanksIterator)
BB_ISOLATED_PAWN = _init_pawn_lookup_tables(AdjacentFilesIterator, AllRanksIterator)

class Contribution(IntEnum):
    Negative = -1
    Neutral = 0
    Positive = 1

    def invert(self):
        return Contribution(-1 * self.value)

class ScanBoard(chess.Board):
    pass

    def get_pieces(self, color: chess.Color, piece_types: list = all_pieces) -> chess.SquareSet:
        pieces = chess.SquareSet()
        for pt in piece_types:
            pieces = pieces.union(self.pieces(pt, color))
        return pieces

    def xrays(self, src:chess.Square, dst:chess.Square) -> bool:
        src_piece = self.piece_at(src)
        if not src_piece:
            return False
        mask = chess.SquareSet()
        if src_piece.piece_type in [chess.ROOK, chess.QUEEN]:
            mask |= chess.BB_FILE_ATTACKS[src][0]
            mask |= chess.BB_RANK_ATTACKS[src][0]
        elif src_piece.piece_type in [chess.BISHOP, chess.QUEEN]:
            mask |= chess.BB_DIAG_ATTACKS[src][0]
        blocking_piece_exists = bool(self.occupied & int(ScanSquareSet.between(src, dst)))
        return bool(mask.intersection(chess.SquareSet.from_square(dst))) and blocking_piece_exists

class ScanSquareSet(chess.SquareSet):
    pass

    def to_str(self) -> str:
        ret = []
        for sq in self:
            ret.append(chess.square_name(sq))
        return '_'.join(ret)

def _connected_aligned(board:ScanBoard, sqs:ScanSquareSet) -> ScanSquareSet:
    if len(sqs) < 2:
        return
    target_iterator = iter(sqs)
    previous_target = next(target_iterator)
    print(chess.square_name(previous_target))
    current_target = next(target_iterator)
    print(chess.square_name(current_target))
    process = True
    chain = [previous_target]
    while process:
        interval = ScanSquareSet.between(previous_target, current_target)
        obstacle = int(interval) & board.occupied
        if obstacle:
            if len(chain) > 1:
                yield ScanSquareSet(chain)
            chain = [current_target]
        else:
            chain.append(current_target)
        previous_target = current_target
        try:
            current_target = next(target_iterator)
        except StopIteration:
            if len(chain) > 1:
                yield ScanSquareSet(chain)
            process = False

class PositionScanner:
    def __init__(self) -> None:
        logging.debug("BEGIN - Building set of features")
        self.features = [\
            UndefendedPiece(),\
            OverloadedPiece(),\
            AbsolutePin(),\
            RelativePin(),\
            XRay(),\
            MajorPieceAlignment(),\
            ForkableMajorPieces(),\
            DoubledPawns(),\
            IsolatedPawn(),\
            PassedPawn(),\
            BackwardPawn(),\
            WeakSquare(),\
            Outpost(),\
            KingInCheck(),\
            KingAsOnlyDefender(),\
            BankRankWeakness(),\
            RestrictedKingMobility(),\
            BishopPair(),\
        ]
        logging.debug("END - Building set of features")

    def list_features(self) -> dict:
        features = {}
        for feature in self.features:
            features[feature.shortcut()] = {'id': feature.id(), 'name': feature.name(), 'description': feature.description(), 'shortcut': feature.shortcut()}
        return features

    def analyze(self, board: ScanBoard) -> dict:
        arrow_colors = {\
            'arrow green': '#00ff0080',\
            'arrow red': '#ff000080',\
            'arrow blue': '#0000ff80',\
            'arrow yellow': '#ffee0080',\
        }
        results = {Contribution.Positive: {}, Contribution.Neutral: {}, Contribution.Negative: {}}
        logging.debug("BEGIN - Scanning position")
        for feature in self.features:
            feature.scan(board)
            non_empty_feature = bool(feature.instances[chess.WHITE]) | bool(feature.instances[chess.BLACK])
            if not non_empty_feature:
                logging.debug("EMPTY feature %s", feature.id())
                continue
            feature_id = feature.id()
            for color in [chess.WHITE, chess.BLACK]:
                contribution = self.get_contribution(board.turn, color, feature.contribution())
                feature_entry = results[contribution].setdefault(feature_id, {})
                color_entry = feature_entry.setdefault('color', {})
                square_to_instance = color_entry.setdefault('square_to_instance', {})
                instances = color_entry.setdefault('instances', {})
                for instance_str, instance in feature.instances[color].items():
                    main_square_set = instance.main
                    aux_square_set = instance.aux
                    check = instance.check
                    arrows = instance.arrows
                    main_str = main_square_set.to_str()
                    aux_str = aux_square_set.to_str()
                    svg_str = chess.svg.board(board, orientation=board.turn, arrows=arrows, check=check, colors=arrow_colors) if check\
                        else chess.svg.board(board, orientation=board.turn, arrows=arrows, colors=arrow_colors)
                    for sq in main_square_set:
                        square_to_instance[chess.square_name(sq)] = main_str
                    main_squares = {chess.square_name(sq): False for sq in main_square_set}
                    instances[main_str] = {'color': color, 'main': main_str, 'aux': aux_str, 'svg': svg_str, 'main_squares': main_squares, 'remaining': len(main_squares)}
        logging.debug("END - Scanning position")
        return results

    def get_contribution(self, turn: chess.Color, color: chess.Color, feature_contribution: Contribution) -> Contribution:
        ret = Contribution.Neutral
        if feature_contribution != Contribution.Neutral:
            ret = feature_contribution if color == turn else feature_contribution.invert()
        return ret

    def square_shortcut(self, square: chess.Square) -> str:
        return ' '.join([c.lower() for c in chess.square_name(square)])

class FeatureInstance:
    def __init__(self, color: chess.Color, contribution:Contribution, main: ScanSquareSet, aux:ScanSquareSet, check: chess.Square = None, arrows:list = []) -> None:
        self._color = color
        self._main = main
        self._aux = aux
        self._check = check
        self._arrows = self.circle_squares(main, contribution)
        self._arrows += self.circle_squares(aux, contribution, color='yellow')
        if arrows:
            self._arrows += arrows

    @property
    def color(self) -> chess.Color:
        return self._color

    @property
    def main(self) -> ScanSquareSet:
        return self._main

    @property
    def aux(self) -> ScanSquareSet:
        return self._aux

    @property
    def check(self) -> chess.Square:
        return self._check

    @property
    def arrows(self) -> list:
        return self._arrows

    def color_by_contribution(self, contribution:Contribution) -> str:
        return 'green' if contribution == Contribution.Positive else\
            ('red' if contribution == Contribution.Negative else 'blue')

    def circle_squares(self, squares: ScanSquareSet, contribution:Contribution, color:str = None) -> list:
        arrow_color = self.color_by_contribution(contribution) if not color else color
        return [chess.svg.Arrow(sq, sq, color=arrow_color) for sq in squares]

class Feature(ABC):
    def __init__(self) -> None:
        self.instances = {}
        self.instances[chess.WHITE] = {}
        self.instances[chess.BLACK] = {}

    def scan(self, board: ScanBoard):
        logging.debug("BEGIN - Scanning %s for WHITE", self.id())
        self.scan_color(board, chess.WHITE)
        logging.debug("END - Scanning %s for WHITE", self.id())
        logging.debug("BEGIN - Scanning %s for BLACK", self.id())
        self.scan_color(board, chess.BLACK)
        logging.debug("END - Scanning %s for BLACK", self.id())
    
    def add_instance(self, color: chess.Color, main: ScanSquareSet, aux: ScanSquareSet, check:chess.Square = None, arrows:list = []) -> None:
        main_str = main.to_str()
        self.instances[color][main_str] = FeatureInstance(color, self.contribution(), main, aux, check, arrows)
        logging.debug("%s: %s", self.id(), main_str)

    def shortcut(self) -> str:
        return ' '.join([c.lower() for c in self.id()])

    @property
    @abstractmethod
    def id(self) -> str:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @property
    @abstractmethod
    def contribution(self) -> Contribution:
        pass

    @abstractmethod
    def scan_color(self, color: chess.Color):
        pass

class UndefendedPiece(Feature):
    def __init__(self) -> None:
        super().__init__()

    def id(self) -> str:
        return 'UDF'

    def name(self) -> str:
        return 'Undefended piece'

    def description(self) -> str:
        return 'Undefended knight, bishop, rook or queen.'

    def contribution(self) -> Contribution:
        return Contribution.Negative

    def scan_color(self, board: ScanBoard, color: chess.Color) -> None:
        main_square_set = ScanSquareSet()
        squares_to_scan = board.get_pieces(color, piece_types=all_pieces)
        for sq in squares_to_scan:
            if not board.is_attacked_by(color, sq):
                main_square_set.add(sq)
        if main_square_set:
            self.add_instance(color, main_square_set, ScanSquareSet())

class OverloadedPiece(Feature):
    def __init__(self) -> None:
        super().__init__()

    def id(self) -> str:
        return 'OVL'

    def name(self) -> str:
        return 'Overloaded piece'

    def description(self) -> str:
        return 'More than one piece have another piece as their unique defender. That is, if such overloaded piece is eliminated, another one becomes undefended.'

    def contribution(self) -> Contribution:
        return Contribution.Negative

    def scan_color(self, board: ScanBoard, color: chess.Color) -> None:
        pass

class AbsolutePin(Feature):
    def __init__(self) -> None:
        super().__init__()

    def id(self) -> str:
        return 'APN'

    def name(self) -> str:
        return 'Absolute pin'

    def description(self) -> str:
        return 'Piece which can not move because doing so would reveal an attack on the king.'

    def contribution(self) -> Contribution:
        return Contribution.Negative

    def scan_color(self, board: ScanBoard, color: chess.Color) -> None:
        king_square = board.king(color)
        squares_to_scan = board.get_pieces(color, piece_types=all_pieces_and_pawns)
        for sq in squares_to_scan:
            if board.is_pinned(color, sq):
                direction = board.pin(color, sq)
                long_range_attackers = board.get_pieces(not color, piece_types=long_range_pieces)
                king_attackers = board.attackers(not color, king_square)
                attackers = long_range_attackers.intersection(direction)
                aux_squares = []
                for a in attackers:
                    if sq in ScanSquareSet.between(king_square, a):
                        aux_squares.append(a)
                # TODO: When len(aux_squares) > 1, select only the closer attacker to the king
                arrows = [chess.svg.Arrow(a, king_square, color='red') for a in aux_squares]
                main_square_set = ScanSquareSet([sq])
                self.add_instance(color, main_square_set, ScanSquareSet(aux_squares), False, arrows)
                logging.debug("%s: %s", self.id(), main_square_set.to_str())


class RelativePin(Feature):
    def __init__(self) -> None:
        super().__init__()

    def id(self) -> str:
        return 'RPN'

    def name(self) -> str:
        return 'Relative pin'

    def description(self) -> str:
        return 'Piece which, if moved, allows the opponent to gain a winning advantage (typically in the form of a significant material loss).'

    def contribution(self) -> Contribution:
        return Contribution.Negative

    def scan_color(self, board: ScanBoard, color: chess.Color) -> None:
        pass

class MajorPieceAlignment(Feature):
    def __init__(self) -> None:
        super().__init__()

    def id(self) -> str:
        return 'MPA'

    def name(self) -> str:
        return 'Major piece alignment'

    def description(self) -> str:
        return 'Rooks, queen or king are in the same rank, file or diagonal.'

    def contribution(self) -> Contribution:
        return Contribution.Negative

    def scan_color(self, board: ScanBoard, color: chess.Color) -> None:
        squares_to_scan = board.get_pieces(color, piece_types=[chess.ROOK, chess.QUEEN, chess.KING])
        alignment = {}
        for sq1 in squares_to_scan:
            for sq2 in squares_to_scan:
                if sq1 == sq2:
                    continue
                ray = chess.SquareSet.ray(sq1, sq2)
                if ray:
                    entry = alignment.setdefault(int(ray), ScanSquareSet())
                    entry.add(sq1)
                    entry.add(sq2)
        for ray, sqs in alignment.items():
            for main_square_set in _connected_aligned(board, sqs):
                self.add_instance(color, main_square_set, ScanSquareSet())
                logging.debug("%s: %s", self.id(), main_square_set.to_str())

class ForkableMajorPieces(Feature):
    def __init__(self) -> None:
        super().__init__()

    def id(self) -> str:
        return 'FMP'

    def name(self) -> str:
        return 'Forkable major pieces'

    def description(self) -> str:
        return 'Rooks, queen or king are can be forked from an undefended square.'

    def contribution(self) -> Contribution:
        return Contribution.Negative

    def scan_color(self, board: ScanBoard, color: chess.Color) -> None:
        squares_to_scan = board.get_pieces(color, piece_types=[chess.ROOK, chess.QUEEN, chess.KING])
        for sq1 in squares_to_scan:
            mask1 = ScanSquareSet(chess.BB_KNIGHT_ATTACKS[sq1])
            for sq2 in squares_to_scan:
                if sq1 == sq2:
                    continue
                mask2 = ScanSquareSet(chess.BB_KNIGHT_ATTACKS[sq2])
                forking_squares = ScanSquareSet(mask1.intersection(mask2))
                if forking_squares:
                    self.add_instance(color, forking_squares, ScanSquareSet([sq1, sq2]))
                    logging.debug("%s: %s", self.id(), forking_squares.to_str())

class BankRankWeakness(Feature):
    def __init__(self) -> None:
        super().__init__()

    def id(self) -> str:
        return 'BRK'

    def name(self) -> str:
        return 'Bank rank weakness'

    def description(self) -> str:
        return 'A defender piece must stay on the bank rank in order to prevent mate.'

    def contribution(self) -> Contribution:
        return Contribution.Negative

    def scan_color(self, board: ScanBoard, color: chess.Color) -> None:
        pass

class BackwardPawn(Feature):
    def __init__(self) -> None:
        super().__init__()

    def id(self) -> str:
        return 'BKP'

    def name(self) -> str:
        return 'Backward pawn'

    def description(self) -> str:
        return 'Pawn that can not be supported by adjacent pawns (because they have advanced ahead of the backward pawn or because they no longer exist).'

    def contribution(self) -> Contribution:
        return Contribution.Negative

    def scan_color(self, board: ScanBoard, color: chess.Color) -> None:
        own_pawns = board.get_pieces(color, [chess.PAWN])
        main_square_set = ScanSquareSet()
        for pawn in own_pawns:
            mask = ScanSquareSet(BB_BACKWARD_PAWN[color][int(ScanSquareSet([pawn]))])
            if len(mask.intersection(own_pawns)) == 1:
                main_square_set.add(pawn)
        if main_square_set:
            self.add_instance(color, main_square_set, ScanSquareSet())
            logging.debug("%s: %s", self.id(), main_square_set.to_str())

class IsolatedPawn(Feature):
    def __init__(self) -> None:
        super().__init__()

    def id(self) -> str:
        return 'ISP'

    def name(self) -> str:
        return 'Isolated pawn'

    def description(self) -> str:
        return 'Pawn without other pawns in the adjacent files.'

    def contribution(self) -> Contribution:
        return Contribution.Neutral

    def scan_color(self, board: ScanBoard, color: chess.Color) -> None:
        pawns = board.get_pieces(color, [chess.PAWN])
        main_square_set = ScanSquareSet()
        for pawn in pawns:
            mask = ScanSquareSet(BB_ISOLATED_PAWN[color][int(ScanSquareSet([pawn]))])
            if len(mask.intersection(pawns)) == 1:
                main_square_set.add(pawn)
        if main_square_set:
            self.add_instance(color, main_square_set, ScanSquareSet())
            logging.debug("%s: %s", self.id(), main_square_set.to_str())

class DoubledPawns(Feature):
    def __init__(self) -> None:
        super().__init__()

    def id(self) -> str:
        return 'DBP'

    def name(self) -> str:
        return 'Doubled pawns'

    def description(self) -> str:
        return 'More than one pawn stacked in the same file.'

    def contribution(self) -> Contribution:
        return Contribution.Negative

    def scan_color(self, board: ScanBoard, color: chess.Color) -> None:
        own_pawns = board.get_pieces(color, [chess.PAWN])
        for f in chess.BB_FILES:
            file_mask = ScanSquareSet(f)
            pawns_in_file = ScanSquareSet(file_mask.intersection(own_pawns))
            if len(pawns_in_file) > 1:
                self.add_instance(color, pawns_in_file, ScanSquareSet())
                logging.debug("%s: %s", self.id(), pawns_in_file.to_str())

class PassedPawn(Feature):
    def __init__(self) -> None:
        super().__init__()

    def id(self) -> str:
        return 'PSP'

    def name(self) -> str:
        return 'Passed pawn'

    def description(self) -> str:
        return 'Pawn that can reach promotion without being challenged by enemy pawns.'

    def contribution(self) -> Contribution:
        return Contribution.Positive

    def scan_color(self, board: ScanBoard, color: chess.Color) -> None:
        own_pawns = board.get_pieces(color, [chess.PAWN])
        opponent_pawns = board.get_pieces(not color, [chess.PAWN])
        main_square_set = ScanSquareSet()
        for pawn in own_pawns:
            mask = ScanSquareSet(BB_PASSED_PAWN[color][int(ScanSquareSet([pawn]))])
            if not mask.intersection(opponent_pawns):
                main_square_set.add(pawn)
        if main_square_set:
            self.add_instance(color, main_square_set, ScanSquareSet())
            logging.debug("%s: %s", self.id(), main_square_set.to_str())

class XRay(Feature):
    def __init__(self) -> None:
        super().__init__()

    def id(self) -> str:
        return 'XRY'

    def name(self) -> str:
        return 'X-Ray'

    def description(self) -> str:
        return 'Enemy piece that can be subject to a discovery attack.'

    def contribution(self) -> Contribution:
        return Contribution.Positive

    def scan_color(self, board: ScanBoard, color: chess.Color) -> None:
        targets = board.get_pieces(not color, piece_types=[chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN])
        long_range_attackers = board.get_pieces(color, piece_types=long_range_pieces)
        for attacker in long_range_attackers:
            xrayed_squares = ScanSquareSet()
            for target in targets:
                if board.xrays(attacker, target):
                    xrayed_squares.add(target)
            if xrayed_squares:
                main_square_set = ScanSquareSet.from_square(attacker)
                self.add_instance(color, main_square_set, xrayed_squares)
                logging.debug("%s: %s", self.id(), main_square_set.to_str())

class RestrictedKingMobility(Feature):
    def __init__(self) -> None:
        super().__init__()

    def id(self) -> str:
        return 'RKM'

    def name(self) -> str:
        return 'Restricted king mobility'

    def description(self) -> str:
        return 'King with zero or one legal move.'

    def contribution(self) -> Contribution:
        return Contribution.Negative

    def scan_color(self, board: ScanBoard, color: chess.Color) -> None:
        # Pretend it´s color´s turn, so we can use local_board.find_move()
        local_board = copy.deepcopy(board)
        if color != local_board.turn:
            local_board.turn = color
        king_square = local_board.king(color)
        squares_to_scan = ScanSquareSet(chess.BB_KING_ATTACKS[king_square])
        king_legal_moves = 0
        legal_squares = ScanSquareSet()
        for sq in squares_to_scan:
            if local_board.is_legal(chess.Move(king_square, sq)):
                legal_squares.add(sq)
                king_legal_moves += 1
                if king_legal_moves == 2:
                    break
        if king_legal_moves < 2:
            main_square_set = ScanSquareSet([king_square])
            self.add_instance(color, main_square_set, legal_squares)
            logging.debug("%s: %s", self.id(), main_square_set.to_str())

class WeakSquare(Feature):
    def __init__(self) -> None:
        super().__init__()

    def id(self) -> str:
        return 'WSQ'

    def name(self) -> str:
        return 'Weak square'

    def description(self) -> str:
        return 'Square that can no longer be defended by pawns.'

    def contribution(self) -> Contribution:
        return Contribution.Negative

    def scan_color(self, board: ScanBoard, color: chess.Color) -> None:
        pass

class Outpost(Feature):
    def __init__(self) -> None:
        super().__init__()

    def id(self) -> str:
        return 'OPS'

    def name(self) -> str:
        return 'Outpost'

    def description(self) -> str:
        return 'Square that, if occupied by a piece, can not be forced away by pawns.'

    def contribution(self) -> Contribution:
        return Contribution.Positive

    def scan_color(self, board: ScanBoard, color: chess.Color) -> None:
        pass

class KingAsOnlyDefender(Feature):
    def __init__(self) -> None:
        super().__init__()

    def id(self) -> str:
        return 'KOD'

    def name(self) -> str:
        return 'King as only defender'

    def description(self) -> str:
        return 'Pawn, knight, bishop, rook or queen defended only by the king.'

    def contribution(self) -> Contribution:
        return Contribution.Negative

    def scan_color(self, board: ScanBoard, color: chess.Color) -> None:
        king_square = board.king(color)
        squares_to_scan = board.get_pieces(color, piece_types=all_pieces_and_pawns)
        main_squares = []
        for sq in squares_to_scan:
            attackers = board.attackers(color, sq)
            if len(attackers) == 1 and king_square in attackers:
                main_squares.append(sq)
        if main_squares:
            main_square_set = ScanSquareSet(main_squares)
            self.add_instance(color, main_square_set, ScanSquareSet([king_square]))
            logging.debug("%s: %s", self.id(), main_square_set.to_str())

class KingInCheck(Feature):
    def __init__(self) -> None:
        super().__init__()

    def id(self) -> str:
        return 'CHK'

    def name(self) -> str:
        return 'King is in check'

    def description(self) -> str:
        return 'One or more opponent pieces are giving check.'

    def contribution(self) -> Contribution:
        return Contribution.Negative

    def scan_color(self, board: ScanBoard, color: chess.Color) -> None:
        if color == board.turn and board.is_check():
            king_square = board.king(color)
            checkers = board.checkers()
            arrows = [chess.svg.Arrow(c, king_square, color='red') for c in checkers]
            main_square_set = ScanSquareSet(chess.SquareSet(checkers))
            self.add_instance(color, main_square_set, ScanSquareSet([king_square]), king_square, arrows)
            logging.debug("%s: %s", self.id(), main_square_set.to_str())

class BishopPair(Feature):
    def __init__(self) -> None:
        super().__init__()

    def id(self) -> str:
        return 'BPR'

    def name(self) -> str:
        return 'Bishop pair'

    def description(self) -> str:
        return 'Only one side still has the two bishops.'

    def contribution(self) -> Contribution:
        return Contribution.Positive

    def scan_color(self, board: ScanBoard, color: chess.Color) -> None:
        own_bishops = board.get_pieces(color, piece_types=[chess.BISHOP])
        opponent_bishops = board.get_pieces(not color, piece_types=[chess.BISHOP])
        if len(own_bishops) == 2 and len(opponent_bishops) < 2:
            main_square_set = ScanSquareSet(own_bishops)
            self.add_instance(color, main_square_set, ScanSquareSet(opponent_bishops))
            logging.debug("%s: %s", self.id(), main_square_set.to_str())
