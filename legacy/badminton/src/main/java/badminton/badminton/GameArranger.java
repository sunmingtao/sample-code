package badminton.badminton;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class GameArranger {
    private final static int PLAYER_NUMBER_ON_ONE_COURT = 4;
    private final static int DEFAULT_GAME_NUMBER = 10;
    private final List<Player> players;
    private final int courtNumber;
    private final List<Game> games;

    public GameArranger(int playerNumber, int courtNumber, int gameNumber) {
        super();
        players = new ArrayList<Player>(playerNumber);
        //Note: player starts id=1, NOT 0
        for (int i = 1; i <= playerNumber; i++) {
            players.add(new Player(i));
        }
        games = new ArrayList<Game>(gameNumber);
        for (int i = 1; i <= gameNumber; i++) {
            games.add(new Game(i));
        }
        this.courtNumber = courtNumber;
    }

    public GameArranger(int playerNumber, int courtNumber) {
        this(playerNumber, courtNumber, DEFAULT_GAME_NUMBER);
    }

    /**
     * Compiles player rotation
     */
    public void arrange() {
        for (Game game : games) {
            game.addCourts(selectPlayersForCourts());
            checkFairness();
        }
    }

    /**
     * Checks fairness (e.g. whether one player has played 3 games while another player
     * has just played 1 game)
     * FIXME Only for debugging purpose, will remove in PROD
     */
    private void checkFairness() {
        int minCount = getMinCount(players);
        int maxCount = getMaxCount(players);
        if (maxCount - minCount > 1) {
            throw new RuntimeException("Not fair! some players already played " + maxCount
                    + " games while some players just played " + minCount + " games");
        }
    }

    /**
     * Selects players for all the courts
     *
     * @return a list of courts with players selected
     */
    private List<Court> selectPlayersForCourts() {
        List<Court> courts = new ArrayList<Court>();
        for (int i = 1; i <= courtNumber; i++) {
            courts.add(new Court(i));
        }
        List<Player> remainingPlayers = copy(players);
        for (Court court : courts) {
            List<Player> players = selectPlayers(PLAYER_NUMBER_ON_ONE_COURT, remainingPlayers);
            court.addPlayers(players);
            remainingPlayers.removeAll(players);
        }
        if (PLAYER_NUMBER_ON_ONE_COURT * courts.size() + remainingPlayers.size() != players.size()) {
            //Should never reach here
            //FIXME Just for debugging purpose, will remove in PROD
            throw new RuntimeException
                    ("Selected number of players + remaining number of players does not equal to total number of players");
        }
        return courts;
    }

    /**
     * Select the specified number of players from a list of players
     *
     * @param players   A list of available players
     * @param playerNum number of players to be selected
     * @return
     */
    private List<Player> selectPlayers(int playerNum, List<Player> players) {
        if (playerNum > players.size()) {
            throw new IllegalArgumentException
                    ("The number of player to be selected exceeds the total number of players available");
        }
        List<Player> selectedPlayers = new ArrayList<Player>();
        List<Player> remainingPlayers = copy(players);
        for (int i = 0; i < playerNum; i++) {
            //Apply filter here
            List<Player> candidates = getCandidates(remainingPlayers);
            Player player = selectRandomPlayer(candidates);
            player.incrementGameNumberPlayed();
            selectedPlayers.add(player);
            remainingPlayers.remove(player);
        }
        if (selectedPlayers.size() + remainingPlayers.size() != players.size()) {
            //Should never reach here
            //FIXME Just for debugging purpose, will remove in PROD
            throw new RuntimeException
                    ("Selected number of players + remaining number of players does not equal to total number of players");
        }
        return selectedPlayers;
    }

    /**
     * Returns a random player from a specified list of players
     *
     * @param players
     * @return
     */
    private Player selectRandomPlayer(List<Player> players) {
        int randomNumber = new Random().nextInt(players.size());
        return players.get(randomNumber);
    }

    /**
     * Returns a list of candidate players from a specified list of players
     * candidates are picked based on follwing criteria
     * 1. The players who have played the fewest number of games
     * 2. TODO The players who have played the fewest times with
     * the already selected players on this court
     * 3. TODO The players who have the closest skill level with
     * the already selected players on this court
     *
     * @param players
     * @return
     */
    private List<Player> getCandidates(List<Player> players) {
        List<Player> candidates = getMinCountPlayers(players);
        //apply more filters later....
        return candidates;
    }

    /**
     * Returns the smallest number of games played among the specified list of players
     *
     * @param players
     * @return
     */
    private static int getMinCount(List<Player> players) {
        int minCount = Integer.MAX_VALUE;
        for (Player player : players) {
            if (player.getCount() < minCount) {
                minCount = player.getCount();
            }
        }
        return minCount;
    }

    /**
     * Returns the greatest number of games played among the specified list of players
     *
     * @param players
     * @return
     */
    private static int getMaxCount(List<Player> players) {
        int maxCount = -1;
        for (Player player : players) {
            if (player.getCount() > maxCount) {
                maxCount = player.getCount();
            }
        }
        return maxCount;
    }

    /**
     * Returns a list players who have played the fewest number of games
     *
     * @param players
     * @return
     */
    private static List<Player> getMinCountPlayers(List<Player> players) {
        List<Player> list = new ArrayList<Player>();
        int minCount = getMinCount(players);
        for (Player player : players) {
            if (player.getCount() == minCount) {
                list.add(player);
            }
        }
        return list;
    }

    /**
     * Makes a shallow copy of list of players
     *
     * @param players
     * @return
     */
    private static List<Player> copy(List<Player> players) {
        List<Player> list = new ArrayList<Player>();
        for (Player player : players) {
            list.add(player);
        }
        return list;
    }

    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (Game game : games) {
            sb.append("Game ID: " + game.getId()).append("\n");
            sb.append(game.toString()).append("\n");
        }
        return sb.toString();
    }
}
