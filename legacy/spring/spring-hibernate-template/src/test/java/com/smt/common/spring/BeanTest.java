package com.smt.common.spring;

import junit.framework.Assert;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.test.context.ContextConfiguration;
import org.springframework.test.context.junit4.AbstractTransactionalJUnit4SpringContextTests;
import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;

import com.smt.common.spring.hibernate.dao.HibernateDao;
import com.smt.common.spring.hibernate.entity.Person;

@RunWith(SpringJUnit4ClassRunner.class)
@ContextConfiguration(locations = {"classpath:context/context.xml"})
public class BeanTest extends AbstractTransactionalJUnit4SpringContextTests{

	@Autowired
	private HibernateDao dao;
	
	@Before
	public final void setup() {
		executeSqlScript("schema.sql", false);
        executeSqlScript("test-data.sql", false);
	}
	
	@Test
	public void testContext(){
		Person p = new Person();
		p.setName("abc");
		dao.save(p);
		Person p2 = dao.findPersonByName("abc");
		Assert.assertEquals(p2.getName(), "abc");
	}
}
