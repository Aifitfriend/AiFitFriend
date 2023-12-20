package com.capstone.fitfriendapp.view.profile

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.Button
import com.capstone.fitfriendapp.R
import com.capstone.fitfriendapp.databinding.ActivityEditProfileBinding

class EditProfileActivity : AppCompatActivity() {

    private lateinit var binding: ActivityEditProfileBinding
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityEditProfileBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val btnCancel: Button = findViewById(R.id.button_cancel)
        btnCancel.setOnClickListener {
            val moveIntent = Intent(this@EditProfileActivity, ProfileActivity::class.java)
            startActivity(moveIntent)
        }

        val btnSave: Button = findViewById(R.id.button_save)
        btnSave.setOnClickListener {
            val moveIntent = Intent(this@EditProfileActivity, ProfileActivity::class.java)
            startActivity(moveIntent)
        }
    }
}