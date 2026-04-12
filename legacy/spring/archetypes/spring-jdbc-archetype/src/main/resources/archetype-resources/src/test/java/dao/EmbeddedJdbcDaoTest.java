package ${package}.dao;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.test.context.ContextConfiguration;
import org.springframework.test.context.junit4.AbstractTransactionalJUnit4SpringContextTests;
import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;


import static junit.framework.Assert.*;

@RunWith(SpringJUnit4ClassRunner.class)
@ContextConfiguration(locations = {"classpath:context/test-context.xml"})
public class EmbeddedJdbcDaoTest extends AbstractTransactionalJUnit4SpringContextTests{

	@Test
	public void testContext(){
		int id = simpleJdbcTemplate.queryForInt("select t.ID from T_PERSON t where t.name = ?", 
				"Mingtao");
		assertEquals(1, id);
	}
}
