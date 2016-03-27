package ${package}.dao;

import ${package}.dao.impl.Person;

public interface JdbcDao {
	public void insert(int id, String name);
	
	public void update(int id, String newName);
	
	public Person select(int id);
	
	public void delete(int id);
}
