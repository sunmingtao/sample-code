package ${package}.dao;

import static junit.framework.Assert.assertEquals;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.test.context.ContextConfiguration;
import org.springframework.test.context.junit4.AbstractTransactionalJUnit4SpringContextTests;
import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;

import ${package}.dao.impl.JdbcDaoImpl;
import ${package}.dao.impl.Person;

@RunWith(SpringJUnit4ClassRunner.class)
@ContextConfiguration(locations = {"classpath:context/context.xml"})
public class ExternalJdbcDaoTest extends AbstractTransactionalJUnit4SpringContextTests{

	@Autowired
	private JdbcDaoImpl dao;

	@Before
	public final void setup() {
		executeSqlScript("schema.sql", false);
        executeSqlScript("test-data.sql", false);
	}
	 
	@Test
	public void testInsert(){
		dao.insert(2, "Ming");
		int id = simpleJdbcTemplate.queryForInt("select t.ID from T_PERSON t where t.name = ?", 
				"Ming");
		assertEquals(2, id);
		assertEquals(2, rowCount());
	}
	
	@Test
	public void testDelete(){
		dao.delete(1);
		assertEquals(0, rowCount());
	}
	
	@Test
	public void testUpdate(){
		dao.update(1, "Sun");
		assertEquals(1, rowCount());
		Person p = dao.select(1);
		assertEquals("Sun", p.getName());
	}
	
	@Test
	public void testSelect(){
		Person p = dao.select(1);
		assertEquals("Mingtao", p.getName());
	}
	
	private int rowCount(){
		return simpleJdbcTemplate.queryForInt("select count(*) from T_PERSON");
	}
	
}
