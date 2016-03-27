package com.smt.common.spring.hibernate.dao;

import com.smt.common.spring.hibernate.entity.Person;

public interface HibernateDao {
	public void save(Person person);
	
	public Person findPersonByName(String name);
}
