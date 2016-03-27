package com.baidu.maven.demo;

public final class Customer {
	private String phoneNumber;
	private int transMade;
	
	public Customer(String phoneNumber, int transMade) {
		super();
		this.phoneNumber = phoneNumber;
		this.transMade = transMade;
	}
	public String getPhoneNumber() {
		return phoneNumber;
	}
	public void setPhoneNumber(String phoneNumber) {
		this.phoneNumber = phoneNumber;
	}
	public int getTransMade() {
		return transMade;
	}
	public void setTransMade(int transMade) {
		this.transMade = transMade;
	}
	
	public String toString(){
		return phoneNumber+" "+transMade;
	}
	
}
