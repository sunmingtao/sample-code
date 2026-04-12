package badminton.badminton;

import java.util.ArrayList;
import java.util.List;

public class Court {
    private List<Player> players = new ArrayList<Player>();

    private int id;

    public Court(int id) {
        super();
        this.id = id;
    }

    public int getId() {
        return id;
    }

    public void addPlayer(Player player) {
        this.players.add(player);
    }

    public void addPlayers(List<Player> players) {
        this.players.addAll(players);
    }

    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Court ID: ").append(id).append("\n");
        for (Player player : players) {
            sb.append(player.toString()).append("\n");
        }
        return sb.toString();
    }
}
