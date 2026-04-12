package badminton.badminton;

public class Player {
    private int id;
    /**
     * Number of games played
     */
    private int gameNumberPlayed;

    public Player(int id) {
        super();
        this.id = id;
    }

    public int getId() {
        return id;
    }

    public void setId(int id) {
        this.id = id;
    }

    public int getCount() {
        return gameNumberPlayed;
    }

    public void incrementGameNumberPlayed() {
        this.gameNumberPlayed++;
    }

    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("ID: ").append(id).append("  ");
        sb.append("Number of games played: ").append(gameNumberPlayed);
        return sb.toString();
    }
}
