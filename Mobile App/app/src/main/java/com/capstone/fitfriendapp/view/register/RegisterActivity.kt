package com.capstone.fitfriendapp.view.register

import android.os.Build
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.view.View
import android.view.WindowInsets
import android.view.WindowManager
import android.widget.Toast
import androidx.activity.viewModels
import androidx.appcompat.app.AlertDialog
import com.capstone.fitfriendapp.data.pref.ResultState
import com.capstone.fitfriendapp.databinding.ActivityRegisterBinding
import com.capstone.fitfriendapp.view.ViewModelFactory

class RegisterActivity : AppCompatActivity() {
    private lateinit var binding: ActivityRegisterBinding
    private val viewModel: RegisterViewModel by viewModels {
        ViewModelFactory.getInstance(this)
    }
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityRegisterBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setupView()
        setupAction()
    }

    private fun setupView() {
        @Suppress("DEPRECATION")
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            window.insetsController?.hide(WindowInsets.Type.statusBars())
        } else {
            window.setFlags(
                WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN
            )
        }
        supportActionBar?.hide()
    }

    private fun setupAction() {
        binding.registerButton.setOnClickListener {
            val name = binding.edRegisterName.text.toString()
            val email = binding.edRegisterEmail.text.toString()
            val password = binding.edRegisterPassword.text.toString()

            if (name.isEmpty()) {
                binding.registerName.error = "Nama pengguna tidak boleh kosong"
                return@setOnClickListener
            } else if (email.isEmpty()) {
                binding.registerEmail.error = "Email tidak boleh kosong"
                return@setOnClickListener
            } else if (password.isEmpty()) {
                binding.registerPassword.error = "Password tidak boleh kosong"
                return@setOnClickListener
            }

            viewModel.register(name,email,password).observe(this) { result ->
                when(result) {
                    is ResultState.Loading -> {
                        showLoading(true)
                    }
                    is ResultState.Success -> {
                        showLoading(false)
                        AlertDialog.Builder(this).apply {
                            setTitle("Yeah!")
                            setMessage("Akun dengan $email sudah jadi nih. Yuk, login dan belajar coding.")
                            setPositiveButton("Lanjut") { _, _ ->
                                finish()
                            }
                            create()
                            show()
                        }
                    }
                    is ResultState.Error -> {
                        Toast.makeText(
                            this@RegisterActivity,
                            "gagal melakukan register. Silahkan coba lagi",
                            Toast.LENGTH_SHORT
                        ).show()
                        showLoading(false)
                    }
                }
            }

        }
    }

    private fun showLoading(state: Boolean) {
        binding.progressBar.visibility = if (state) View.VISIBLE else View.GONE
    }

}