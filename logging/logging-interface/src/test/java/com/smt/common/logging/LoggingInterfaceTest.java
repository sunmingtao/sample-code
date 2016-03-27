package com.smt.common.logging;

import org.junit.Test;

public class LoggingInterfaceTest {
	
	@Test
	public void test(){
		LoggingInterface logging = new LoggingInterface();
		logging.log();
	}
}
