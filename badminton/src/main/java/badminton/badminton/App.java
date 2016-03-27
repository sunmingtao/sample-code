package badminton.badminton;

/**
 * Hello world!
 */
public class App {

    private static final int PLAYER_NUMBER = 20;
    private static final int COURT_NUMBER = 2;
    private static final int GAME_NUMBER = 20;

    public static void main(String[] args) {
        //GameArranger playerSelector = new GameArranger(PLAYER_NUMBER, COURT_NUMBER);
        GameArranger playerSelector = new GameArranger(PLAYER_NUMBER, COURT_NUMBER, GAME_NUMBER);
        playerSelector.arrange();
        System.out.println(playerSelector.toString());
    }
}
