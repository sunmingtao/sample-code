package com.smt.common.spring.hibernate.dao.impl;


import org.springframework.orm.hibernate3.support.HibernateDaoSupport;

import com.smt.common.spring.hibernate.dao.HibernateDao;

public abstract class AbstractHibernateDaoImpl<T> extends HibernateDaoSupport implements HibernateDao<T>{

	@Override
	public void save(T t) {
		getHibernateTemplate().save(t);
	}

	@Override
	public void update(T t) {
		getHibernateTemplate().update(t);
	}
	
	@Override
	public void saveOrUpdate(T t) {
		getHibernateTemplate().saveOrUpdate(t);
	}

	@Override
	public void delete(T t) {
		getHibernateTemplate().delete(t);
	}
}
