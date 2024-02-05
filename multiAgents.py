from Agents import Agent
import util
import random

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """
    def __init__(self, *args, **kwargs) -> None:
        self.index = 0 # your agent always has index 0

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        It takes a GameState and returns a tuple representing a position on the game board.
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions(self.index)

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current and proposed successor
        GameStates (Game.py) and returns a number, where higher numbers are better.
        You can try and change this evaluation function if you want but it is not necessary.
        """
        nextGameState = currentGameState.generateSuccessor(self.index, action)
        return nextGameState.getScore(self.index) - currentGameState.getScore(self.index)


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    Every player's score is the number of pieces they have placed on the board.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore(0)


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (Agents.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2', **kwargs):
        self.index = 0 # your agent always has index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(**kwargs)

    def getAction(self, state):
        return self.minimax(state, self.index, 0)[1]

    def minimax(self, state, agent_index, current_depth):
        if current_depth == self.depth or state.isGameFinished():
            return self.evaluationFunction(state), None

        legal_actions = state.getLegalActions(agent_index)
        if not legal_actions:
            return self.evaluationFunction(state), None

        if agent_index == 0:
            max_value = float('-inf')
            best_action = None
            for action in legal_actions:
                successor_state = state.generateSuccessor(agent_index, action)
                value, _ = self.minimax(successor_state, (agent_index + 1) % state.getNumAgents(), current_depth + 1)
                if value > max_value:
                    max_value = value
                    best_action = action
            return max_value, best_action
        else:
            min_value = float('inf')
            for action in legal_actions:
                successor_state = state.generateSuccessor(agent_index, action)
                value, _ = self.minimax(successor_state, (agent_index + 1) % state.getNumAgents(), current_depth)
                min_value = min(min_value, value)
            return min_value, None



class AlphaBetaAgent(MultiAgentSearchAgent):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(**kwargs)

    def getAction(self, state):
        return self.alpha_beta_pruning(state, self.index, 0, float('-inf'), float('inf'))[1]

    def alpha_beta_pruning(self, state, agent_index, current_depth, alpha, beta):
        if current_depth == self.depth or state.isGameFinished():
            return self.evaluationFunction(state), None

        legal_actions = state.getLegalActions(agent_index)
        if not legal_actions:
            return self.evaluationFunction(state), None

        if agent_index == 0:
            max_value = float('-inf')
            best_action = None
            for action in legal_actions:
                successor_state = state.generateSuccessor(agent_index, action)
                value, _ = self.alpha_beta_pruning(successor_state, (agent_index + 1) % state.getNumAgents(),
                                                    current_depth + 1, alpha, beta)
                if value > max_value:
                    max_value = value
                    best_action = action
                alpha = max(alpha, max_value)
                if max_value >= beta:
                    break
            return max_value, best_action
        else:
            min_value = float('inf')
            for action in legal_actions:
                successor_state = state.generateSuccessor(agent_index, action)
                value, _ = self.alpha_beta_pruning(successor_state, (agent_index + 1) % state.getNumAgents(),
                                                    current_depth, alpha, beta)
                min_value = min(min_value, value)
                beta = min(beta, min_value)
                if min_value <= alpha:
                    break
            return min_value, None


class ExpectimaxAgent(MultiAgentSearchAgent):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(**kwargs)

    def getAction(self, state):
        return self.expectimax(state, self.index, 0)[1]

    def expectimax(self, state, agent_index, current_depth):
        if current_depth == self.depth or state.isGameFinished():
            return self.evaluationFunction(state), None

        legal_actions = state.getLegalActions(agent_index)
        if not legal_actions:
            return self.evaluationFunction(state), None

        if agent_index == 0:
            max_value = float('-inf')
            best_action = None
            for action in legal_actions:
                successor_state = state.generateSuccessor(agent_index, action)
                value, _ = self.expectimax(successor_state, (agent_index + 1) % state.getNumAgents(),
                                           current_depth + 1)
                if value > max_value:
                    max_value = value
                    best_action = action
            return max_value, best_action
        else:
            expected_value = 0
            num_actions = len(legal_actions)
            for action in legal_actions:
                successor_state = state.generateSuccessor(agent_index, action)
                value, _ = self.expectimax(successor_state, (agent_index + 1) % state.getNumAgents(),
                                           current_depth)
                expected_value += value / num_actions
            return expected_value, None


def betterEvaluationFunction(currentGameState):

    def calculate_coin_parity():
        player_pieces = currentGameState.getPieces(0)
        opponent_pieces = currentGameState.getPieces(1)

        max_player_coins = len(player_pieces)
        min_player_coins = len(opponent_pieces)

        return 100 * (max_player_coins - min_player_coins) / (max_player_coins + min_player_coins)

    def calculate_actual_mobility():
        max_player_mobility = len(currentGameState.getLegalActions(0))
        min_player_mobility = len(currentGameState.getLegalActions(1))

        if (max_player_mobility + min_player_mobility) != 0:
            return 100 * (max_player_mobility - min_player_mobility) / (max_player_mobility + min_player_mobility)
        else:
            return 0

    def calculate_corners_captured():
        player_corners = len(currentGameState.getCorners())
        opponent_corners = len(currentGameState.getPieces(1))

        if (player_corners + opponent_corners) != 0:
            return 100 * (player_corners - opponent_corners) / (player_corners + opponent_corners)
        else:
            return 0

    def calculate_stability(currentGameState, playerIndex):
        def get_player_stability(currentGameState, playerIndex):
            stable_weight = 1
            unstable_weight = -1
            semi_stable_weight = 0

            player_stability = 0

            player_pieces = currentGameState.getPieces(playerIndex)

            for piece in player_pieces:
                stability_category = get_stability_category(currentGameState, piece, playerIndex)
                if stability_category == "stable":
                    player_stability += stable_weight
                elif stability_category == "unstable":
                    player_stability += unstable_weight
                else:
                    player_stability += semi_stable_weight

            return player_stability

        def get_stability_category(currentGameState, piece, playerIndex):
            # Check if the coin is at a corner
            if piece in currentGameState.getCorners():
                return "stable"

            # Check if the coin is on the edge (but not a corner)
            elif piece[0] == 0 or piece[0] == 7 or piece[1] == 0 or piece[1] == 7:
                return "semi-stable"

            # Check if the coin is surrounded by other same-colored coins
            elif surrounded_by_same_color(currentGameState, piece, playerIndex):
                return "stable"

            # If none of the above conditions are met, the coin is unstable
            else:
                return "unstable"

        def surrounded_by_same_color(currentGameState, piece, playerIndex):
            player_pieces = currentGameState.getPieces(playerIndex)

            cnt = 0
            for p in player_pieces:
                if p == piece:
                    continue
                if abs(piece[0] - p[0]) < 2 and abs(piece[1] - p[1]):
                    cnt = cnt + 1

            return cnt == 8

        max_player_stability = get_player_stability(currentGameState, playerIndex)
        min_player_stability = get_player_stability(currentGameState, 1 - playerIndex)

        if (max_player_stability + min_player_stability) != 0:
            return 100 * (max_player_stability - min_player_stability) / (max_player_stability + min_player_stability)
        else:
            return 0

    # Weights for each heuristic
    coin_parity_weight = 1.0
    actual_mobility_weight = 1.0
    corners_captured_weight = 1.0
    stability_weight = 1.0

    # Calculate each heuristic value
    coin_parity_value = calculate_coin_parity()
    actual_mobility_value = calculate_actual_mobility()
    corners_captured_value = calculate_corners_captured()
    stability_value = calculate_stability(currentGameState, 0)

    # Combine heuristic values with weights to get the final evaluation
    evaluation = (
        coin_parity_weight * coin_parity_value +
        actual_mobility_weight * actual_mobility_value +
        corners_captured_weight * corners_captured_value +
        stability_weight * stability_value
    )

    return evaluation

# Abbreviation
better = betterEvaluationFunction