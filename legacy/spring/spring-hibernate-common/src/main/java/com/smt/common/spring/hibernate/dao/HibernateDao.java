package com.smt.common.spring.hibernate.dao;

public interface HibernateDao<T> {
	public void save(T t);
	
	public void update(T t);
	
	public void saveOrUpdate(T t);
	
	public void delete(T t);
}
