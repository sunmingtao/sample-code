package com.smt.common.logging;

import org.junit.Test;

public class LogbackTest {
	
	@Test
	public void test(){
		LoggingInterface logging = new LoggingInterface();
		logging.log();
	}
}
