package com.smt.common.spring;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.test.context.ContextConfiguration;
import org.springframework.test.context.junit4.AbstractTransactionalJUnit4SpringContextTests;
import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;

import com.smt.common.spring.hibernate.dao.PersonDaoImpl;
import com.smt.common.spring.hibernate.entity.Person;

@RunWith(SpringJUnit4ClassRunner.class)
@ContextConfiguration(locations = {"classpath:context/test-context.xml"})
public class BeanTest extends AbstractTransactionalJUnit4SpringContextTests{

	@Before
	public final void setup() {
		executeSqlScript("sql/schema.sql", false);
        executeSqlScript("sql/test-data.sql", false);
	}
	
	/**
	 * Just to test the context can be loaded OK
	 */
	@Test
	public void testContext(){
		Person p = new Person();
		p.setName("abc");
		PersonDaoImpl dao = applicationContext.getBean(PersonDaoImpl.class);
		dao.save(p);
	}
	
	
}
