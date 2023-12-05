const mongoose = require('mongoose');

// Create schema
const User = mongoose.model('User', {
	uid: {
		type: String,
		required: true,
	},
	email: {
		type: String,
		required: true,
	},
	name: {
		type: String,
		required: true,
	},
	data: {
		gender: {
			type: String,
		},
		height: {
			type: Number,
			default: 0,
		},
		weight: {
			type: Number,
			default: 0,
		},
	},
});

module.exports = User;
