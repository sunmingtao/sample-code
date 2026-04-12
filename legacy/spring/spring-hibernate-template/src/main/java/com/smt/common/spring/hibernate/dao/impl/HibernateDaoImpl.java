package com.smt.common.spring.hibernate.dao.impl;

import java.util.List;

import org.springframework.orm.hibernate3.support.HibernateDaoSupport;

import com.smt.common.spring.hibernate.dao.HibernateDao;
import com.smt.common.spring.hibernate.entity.Person;

public class HibernateDaoImpl extends HibernateDaoSupport implements HibernateDao{
	@Override
	public void save(Person person) {
		getHibernateTemplate().save(person);
	}

	@SuppressWarnings("unchecked")
	@Override
	public Person findPersonByName(String name) {
		List<Person> list = getHibernateTemplate().find(
                "from Person where name=?",name);
		return list.get(0);
	}
}
