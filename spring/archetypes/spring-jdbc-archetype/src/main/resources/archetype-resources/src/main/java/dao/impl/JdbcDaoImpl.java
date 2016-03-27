package ${package}.dao.impl;

import java.sql.ResultSet;
import java.sql.SQLException;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.core.RowMapper;
import org.springframework.stereotype.Repository;

import ${package}.dao.JdbcDao;

@Repository
public class JdbcDaoImpl implements JdbcDao {

	@Autowired
	private JdbcTemplate jdbcTemplate;
	
	@Override
	public void insert(int id, String name) {
		jdbcTemplate.update("insert into T_PERSON values (?, ?)", new Object[]{id, name});
	}

	@Override
	public void update(int id, String newName) {
		jdbcTemplate.update("update T_PERSON set NAME = ? where id = ?", new Object[]{newName, 1});
	}

	@Override
	public Person select(int id) {
		return jdbcTemplate.queryForObject("select * from T_PERSON where id = ?", new Object[]{id}, new PersonMapper());
	}

	private static class PersonMapper implements RowMapper<Person>{
		@Override
		public Person mapRow(ResultSet rs, int rowNum) throws SQLException {
			Person person = new Person();
			person.setId(rs.getInt("ID"));
			person.setName(rs.getString("NAME"));
			return person;
		}
	}
	
	@Override
	public void delete(int id) {
		jdbcTemplate.update("delete from T_PERSON where id = ?", id);

	}

}
