package badminton.badminton;

import java.util.ArrayList;
import java.util.List;

public final class Game {
	private List<Court> courts = new ArrayList<Court>();
	
	private int id;
	
	public Game(int id) {
		super();
		this.id = id;
	}
	
	public int getId() {
		return id;
	}

	public void addCourts(List<Court> courts){
		this.courts = courts;	
	}
	
	public String toString(){
		StringBuilder sb = new StringBuilder();
		for (Court court: courts){
			sb.append(court.toString()).append("\n");
		}
		return sb.toString();
	}
}
