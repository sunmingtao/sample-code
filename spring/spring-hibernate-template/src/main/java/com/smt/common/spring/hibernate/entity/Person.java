package com.smt.common.spring.hibernate.entity;

import javax.persistence.Column;
import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.Id;
import javax.persistence.Table;

@Entity
@Table(name="T_PERSON")
public class Person {
	
	@Id
	@GeneratedValue
	@Column(name="ID")
	private int id;

	@Column(name="NAME")
	private String name;

	public Person(int id, String name) {
		super();
		this.id = id;
		this.name = name;
	}
	
	public Person(){
	}

	public int getId() {
		return id;
	}

	public void setId(int id) {
		this.id = id;
	}

	public String getName() {
		return name;
	}

	public void setName(String name) {
		this.name = name;
	}
	
}
